from pathlib import Path
from src.video_utils import extract_frames
from src.ocr_utils import extract_text_from_image
from src.deepseekocr_utils import extract_text_deepseek
from src.preprocess import clean_ocr_results
from src.io_utils import save_json, load_json, file_exists

_OCR_ENGINES = {
    "easyocr": extract_text_from_image,
    "deepseek": extract_text_deepseek,
}


def process_video(
    video_path: str,
    output_folder: str = "data/frames",
    checkpoint_base_folder: str = "data/ocr_results",
    interval: int = 30,
    clear_frames: bool = True,
    ocr_engine: str = "easyocr",
    resume: bool = True,
    max_frames: int | None = None,
) -> list[dict]:
    """
    Process a video through frame extraction, OCR, preprocessing, and checkpointing.

    Args:
        video_path (str):              Path to the input video.
        output_folder (str):           Base folder for extracted frames.
        checkpoint_base_folder (str):  Base folder for per-frame OCR JSON checkpoints.
        interval (int):                Extract 1 frame every N frames.
        clear_frames (bool):           Delete old extracted frames before running.
        ocr_engine (str):              "easyocr" or "deepseek".
        resume (bool):                 Skip frames whose checkpoint JSON already exists.
        max_frames (int | None):       Stop early after this many frames (useful for testing).

    Returns:
        list[dict]: Per-frame results, each containing:
                    - "frame"        path, frame_index, timestamp_s
                    - "ocr_results"  raw OCR output
                    - "cleaned_text" filtered text strings
                    - "error"        error message string if OCR failed, else None
    """
    if ocr_engine not in _OCR_ENGINES:
        raise ValueError(
            f"Unsupported OCR engine: '{ocr_engine}'. "
            f"Choose from: {list(_OCR_ENGINES.keys())}"
        )

    ocr_fn = _OCR_ENGINES[ocr_engine]
    video_stem = Path(video_path).stem

    scoped_output_folder = f"{output_folder}/{video_stem}_int{interval}"
    checkpoint_folder = f"{checkpoint_base_folder}/{video_stem}_{ocr_engine}_int{interval}"

    frames = extract_frames(
        video_path=video_path,
        output_folder=scoped_output_folder,
        interval=interval,
        clear_existing=clear_frames,
    )

    if max_frames is not None:
        frames = frames[:max_frames]

    print(f"[INFO] Processing {len(frames)} frames with '{ocr_engine}'...")

    all_results = []

    for idx, frame in enumerate(frames):
        frame_path = frame["path"]
        frame_name = Path(frame_path).stem
        ocr_file = f"{checkpoint_folder}/{frame_name}.json"

        # Resume: load existing checkpoint if present
        if resume and file_exists(ocr_file):
            print(f"[RESUME] Loaded checkpoint: {frame_name}")
            frame_result = load_json(ocr_file)
            all_results.append(frame_result)
            continue

        print(f"[INFO] Frame {idx + 1}/{len(frames)}: {frame_path}")

        # Run OCR with per-frame error isolation
        try:
            ocr_results = ocr_fn(frame_path)
            cleaned_text = clean_ocr_results(ocr_results)
            error = None
        except Exception as exc:
            print(f"[WARNING] OCR failed on {frame_path}: {exc}")
            ocr_results = []
            cleaned_text = []
            error = str(exc)

        frame_result = {
            "frame": frame,          # dict: path, frame_index, timestamp_s
            "ocr_results": ocr_results,
            "cleaned_text": cleaned_text,
            "error": error,
        }

        # Only checkpoint successful results to avoid caching failures
        if error is None:
            save_json(frame_result, ocr_file)

        all_results.append(frame_result)

    failed = sum(1 for r in all_results if r.get("error"))
    print(f"[INFO] OCR complete. {len(all_results)} frames processed, {failed} failed.")

    return all_results