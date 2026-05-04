from pathlib import Path
from src.llava_utils import analyse_frame
from src.io_utils import save_json, load_json, file_exists


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
                    - "cleaned_text"         OCR text that was passed to LLaVA
                    - "user_story"           generated user story
                    - "acceptance_criteria"  generated Gherkin criteria
                    - "gap_analysis"         generated gap analysis
                    - "raw_response"         full unstructured model output
                    - "error"                error string if reasoning failed, else None
    """
    Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)

    all_artefacts = []
    total = len(ocr_results)
    skipped_ocr_errors = 0

    for idx, frame_result in enumerate(ocr_results):
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
            print(f"[RESUME] Loaded reasoning checkpoint: {frame_name}")
            all_artefacts.append(load_json(checkpoint_file))
            continue

        print(f"[INFO] Reasoning frame {idx + 1}/{total}: {frame['path']}")

        try:
            artefacts = analyse_frame(
                image_path=frame["path"],
                cleaned_text=frame_result["cleaned_text"],
            )
            error = None
        except Exception as exc:
            print(f"[WARNING] Reasoning failed on {frame['path']}: {exc}")
            artefacts = {
                "user_story": "",
                "acceptance_criteria": "",
                "gap_analysis": "",
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

        all_artefacts.append(result)

    failed = sum(1 for r in all_artefacts if r.get("error"))
    print(
        f"[INFO] Reasoning complete. "
        f"{len(all_artefacts)} frames processed, "
        f"{failed} failed, "
        f"{skipped_ocr_errors} skipped (OCR errors)."
    )

    return all_artefacts