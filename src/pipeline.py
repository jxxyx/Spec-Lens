from pathlib import Path
from src.video_utils import extract_frames
from src.ocr_utils import extract_text_from_image
from src.preprocess import clean_ocr_results
from src.io_utils import save_json, load_json, file_exists


def process_video(
    video_path: str,
    output_folder: str = "data/frames",
    checkpoint_base_folder: str = "data/ocr_results",
    interval: int = 30,
    clear_frames: bool = True,
    ocr_engine: str = "easyocr",
    resume: bool = True,
    max_frames: int | None = None
):
    """
    Process a video through frame extraction, OCR, preprocessing, and checkpointing.

    Args:
        video_path (str): Path to the input video
        output_folder (str): Base folder where extracted frames will be stored
        checkpoint_base_folder (str): Base folder where OCR checkpoint JSON files will be stored
        interval (int): Save 1 frame every 'interval' frames
        clear_frames (bool): Whether to clear old extracted frames before running
        ocr_engine (str): OCR engine label, used for checkpoint naming
        resume (bool): Whether to reuse existing OCR checkpoint files
        max_frames (int | None): Stop early after this many frames, useful for testing resume logic

    Returns:
        list[dict]: OCR + cleaned results for each frame
    """

    video_stem = Path(video_path).stem

    # Create a video-specific frame folder to avoid mixing runs
    scoped_output_folder = f"{output_folder}/{video_stem}_int{interval}"

    # Extract frames
    frames = extract_frames(
        video_path=video_path,
        output_folder=scoped_output_folder,
        interval=interval,
        clear_existing=clear_frames
    )

    print(f"[INFO] Processing {len(frames)} frames with OCR...")

    all_results = []

    # Create a unique checkpoint folder for this video + OCR engine + interval
    checkpoint_folder = f"{checkpoint_base_folder}/{video_stem}_{ocr_engine}_int{interval}"

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

            # For now, only EasyOCR is active here
            # Later, you can add DeepSeek-OCR switching here
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