import cv2
import shutil
from pathlib import Path


def extract_frames(
    video_path: str,
    output_folder: str,
    interval: int = 30,
    clear_existing: bool = True,
) -> list[dict]:
    """
    Extract frames from a video at a fixed frame interval.

    Args:
        video_path (str):      Path to the input video file.
        output_folder (str):   Folder where extracted frames will be saved.
        interval (int):        Save 1 frame every N frames (must be >= 1).
        clear_existing (bool): If True, delete existing output folder before running.

    Returns:
        list[dict]: One entry per saved frame with keys:
                    - "path"        (str)   absolute path to the saved JPEG
                    - "frame_index" (int)   the original frame number in the video
                    - "timestamp_s" (float) timestamp in seconds at the video's FPS
    """
    if interval < 1:
        raise ValueError(f"interval must be >= 1, got {interval}")

    input_path = Path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path = Path(output_folder)

    if clear_existing and output_path.exists():
        shutil.rmtree(output_path)
        print(f"[INFO] Cleared existing frames in: {output_folder}")

    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0  # fallback to 1 if FPS metadata missing

    frame_index = 0
    saved_count = 0
    saved_frames = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % interval == 0:
            frame_file = output_path / f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            saved_frames.append({
                "path": str(frame_file),
                "frame_index": frame_index,
                "timestamp_s": round(frame_index / fps, 3),
            })
            saved_count += 1

        frame_index += 1

    cap.release()

    print(f"[INFO] Frames read: {frame_index} | Frames saved: {saved_count}")
    return saved_frames