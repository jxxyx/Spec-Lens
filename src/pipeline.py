# Core pipeline: video → frames → OCR

from src.video_utils import extract_frames
from src.ocr_utils import extract_text_from_image
from src.preprocess import clean_ocr_results


def process_video(
    video_path: str,
    output_folder: str = "data/frames",
    interval: int = 30,
    clear_frames: bool = True,
    ocr_engine: str = "easyocr"
):
    """
    Process a video and extract OCR text from frames.

    Args:
        video_path (str): Path to video file
        output_folder (str): Where frames are stored
        interval (int): Frame extraction interval
        clear_frames (bool): Whether to clear old frames
        ocr_engine (str): OCR backend (for future extension)

    Returns:
        list[dict]: OCR results per frame
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

    # Step 2: Run OCR on each frame
    for idx, frame in enumerate(frames):
        print(f"[INFO] Processing frame {idx + 1}/{len(frames)}: {frame}")

        ocr_results = extract_text_from_image(frame)
        cleaned_text = clean_ocr_results(ocr_results)

        all_results.append({
            "frame": frame,
            "ocr_results": ocr_results,
            "cleaned_text": cleaned_text
        })

    print("[INFO] OCR processing complete.")

    return all_results