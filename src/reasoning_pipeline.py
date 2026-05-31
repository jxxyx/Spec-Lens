from pathlib import Path
from src.llava_utils import analyse_frame
from src.io_utils import save_json, load_json, file_exists
from src.frame_utils import filter_unique_frames


def run_reasoning(
    ocr_results: list[dict],
    checkpoint_folder: str,
    resume: bool = True,
) -> list[dict]:
    """
    Run the LLaVA reasoning layer over OCR pipeline output.

    Iterates over per-frame OCR results produced by process_video(), feeds each
    frame image + cleaned_text into LLaVA, and returns structured BA artefacts.

    Checkpointing mirrors the OCR pipeline — each frame's reasoning output is
    saved as a JSON file so the run can be resumed if interrupted.

    Args:
        ocr_results (list[dict]):  Output from process_video().
        checkpoint_folder (str):   Folder to store per-frame reasoning JSON files.
        resume (bool):             If True, skip frames with existing checkpoints.

    Returns:
        list[dict]: Per-frame reasoning results, each containing:
                    - "frame"                original frame metadata dict
                    - "cleaned_text"         OCR text passed to LLaVA
                    - "screen_summary"       2–3 sentence screen description
                    - "user_stories"         list[dict] — title, story, scenario per US
                    - "acceptance_criteria"  consolidated Gherkin block (str)
                    - "gap_analysis"         categorised gap bullet list (str)
                    - "truncated"            True if model hit token limit mid-response
                    - "raw_response"         full unstructured model output
                    - "error"                error string if reasoning failed, else None
    """
    Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)

    all_artefacts = []
    total = len(ocr_results)
    skipped_ocr_errors = 0

    # Step 1 - extract flat frame dicts for duplicate detection
    frame_dicts = [item["frame"] for item in ocr_results]

    # Step 2 - filter duplicates
    unique_frames = filter_unique_frames(frame_dicts)

    # Step 3 - build lookup and match back to full ocr_result entries
    lookup = {item["frame"]["frame_index"]: item for item in ocr_results}
    unique_ocr_results = [lookup[frame["frame_index"]] for frame in unique_frames]

    # Step 4 - log how many were deduplicated
    print(f"[DEDUP] {len(ocr_results)} frames → {len(unique_ocr_results)} unique frames")

    # Store summaries of previous unique screens for flow reasoning
    context_window = []

    for idx, frame_result in enumerate(unique_ocr_results):
        frame = frame_result["frame"]
        frame_name = Path(frame["path"]).stem
        checkpoint_file = f"{checkpoint_folder}/{frame_name}.json"

        # Skip frames where OCR itself failed — nothing useful to reason over
        if frame_result.get("error"):
            print(f"[SKIP] Frame {idx + 1}/{total}: OCR error, skipping reasoning.")
            skipped_ocr_errors += 1
            continue

        # Resume: load existing reasoning checkpoint if present
        if resume and file_exists(checkpoint_file):
            loaded = load_json(checkpoint_file)
            all_artefacts.append(loaded)
            
            # Update context window from resumed checkpoint
            if loaded.get("screen_summary"):
                context_window.append(loaded["screen_summary"])
                context_window = context_window[-5:]
            
            continue

        try:
            artefacts = analyse_frame(
                image_path=frame["path"],
                cleaned_text=frame_result["cleaned_text"],
                context_window=context_window,
            )
            error = None
        except Exception as exc:
            print(f"[WARNING] Reasoning failed on {frame['path']}: {exc}")
            artefacts = {
                "screen_summary": "",
                "user_stories": [],
                "acceptance_criteria": "",
                "gap_analysis": "",
                "truncated": False,
                "raw_response": "",
                "image_path": frame["path"],
            }
            error = str(exc)
        

        result = {
            "frame": frame,
            "cleaned_text": frame_result["cleaned_text"],
            **artefacts,
            "error": error,
        }

        # Only checkpoint successful results
        if error is None:
            save_json(result, checkpoint_file)

            # Add the current screen summary into workflow memory
            if artefacts.get("screen_summary"):
                context_window.append(artefacts["screen_summary"])

                # Keep only the latest 5 screen summaries
                context_window = context_window[-5:]


        all_artefacts.append(result)

    failed = sum(1 for r in all_artefacts if r.get("error"))
    print(
        f"[INFO] Reasoning complete. "
        f"{len(all_artefacts)} frames processed, "
        f"{failed} failed, "
        f"{skipped_ocr_errors} skipped (OCR errors)."
    )

    return all_artefacts

