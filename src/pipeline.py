# Core pipeline: video → frames → OCR → preprocessing → checkpoint save/load

from pathlib import Path
from src.video_utils import extract_frames
from src.ocr_utils import extract_text_from_image
from src.preprocess import clean_ocr_results
from src.io_utils import save_json, load_json, file_exists


def process_video(
    video_path: str,
    output_folder: str = "data/frames",
    interval: int = 30,
    clear_frames: bool = True,
    ocr_engine: str = "easyocr",
    resume: bool = True,
    max_frames: int | None = None
):
    """
    Process a video through frame extraction, OCR, and preprocessing.

    Args:
        video_path (str): Path to the video file
        output_folder (str): Folder to save extracted frames
        interval (int): Save 1 frame every 'interval' frames
        clear_frames (bool): Whether to clear old frames before extraction
        ocr_engine (str): OCR engine label for checkpoint folder naming
        resume (bool): If True, reuse saved OCR checkpoint files
        max_frames (int | None): If set, stop after this many frames for resume testing

    Returns:
        list[dict]: OCR + cleaned results for each processed frame
    """

    # Step 1: Extract frames
    frames = extract_frames(
        video_path=video_path,
        output_folder=output_folder,
        interval=interval,
        clear_existing=clear_frames
    )

    print(f"[INFO] Processing {len(frames)} frames with OCR...")

    all_results = []
    video_stem = Path(video_path).stem
    checkpoint_folder = f"data/ocr_results/{video_stem}_{ocr_engine}_int{interval}"

    # Step 2: OCR + preprocessing + checkpointing
    for idx, frame in enumerate(frames):
        if max_frames is not None and idx >= max_frames:
            print(f"[INFO] Stopping early after {max_frames} frames for resume testing.")
            break

        frame_name = Path(frame).stem
        ocr_file = f"{checkpoint_folder}/{frame_name}.json"

        if resume and file_exists(ocr_file):
            print(f"[INFO] Loading existing OCR result for {frame_name}")
            frame_result = load_json(ocr_file)
        else:
            print(f"[INFO] Processing frame {idx + 1}/{len(frames)}: {frame}")

            ocr_results = extract_text_from_image(frame)
            cleaned_text = clean_ocr_results(ocr_results)

            frame_result = {
                "frame": frame,
                "ocr_results": ocr_results,
                "cleaned_text": cleaned_text
            }

            save_json(frame_result, ocr_file)

        all_results.append(frame_result)

    print("[INFO] OCR processing complete.")
    return all_results