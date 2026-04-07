# This script handles video input and transforms it into frames

import cv2
import shutil
from pathlib import Path


def extract_frames(
    video_path: str,
    output_folder: str,
    interval: int = 30,
    clear_existing: bool = True
):
    """
    Extract frames from a video at a fixed interval.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Folder where extracted frames will be saved.
        interval (int): Save 1 frame every 'interval' frames.
        clear_existing (bool): If True, delete existing output folder contents first.

    Returns:
        list[str]: List of saved frame file paths.
    """

    # Convert folder string into a Path object
    output_path = Path(output_folder)

    # Clear old frames if requested
    if clear_existing and output_path.exists():
        shutil.rmtree(output_path)
        print(f"[INFO] Cleared existing frames in: {output_folder}")

    # Recreate output folder
    output_path.mkdir(parents=True, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    # Validate interval
    if interval <= 0:
        cap.release()
        raise ValueError("Interval must be greater than 0.")

    frame_count = 0      # total frames read
    saved_count = 0      # total frames saved
    saved_frames = []    # list of saved frame paths

    # Read video frame by frame
    while True:
        success, frame = cap.read()

        # Stop if video ends
        if not success:
            break

        # Save every X frames
        if frame_count % interval == 0:
            frame_file = output_path / f"frame_{saved_count}.jpg"

            # Save frame to disk
            cv2.imwrite(str(frame_file), frame)

            # Store path for later pipeline use
            saved_frames.append(str(frame_file))
            saved_count += 1

        frame_count += 1

    # Release video memory
    cap.release()

    print(f"[INFO] Total frames read: {frame_count}")
    print(f"[INFO] Total frames saved: {saved_count}")

    return saved_frames