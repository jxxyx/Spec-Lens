# Load the libraries required
import cv2
import numpy as np


# Function to check for duplication of frames via pixel difference
def is_duplicate(frame_path_A: str, frame_path_B: str, threshold: float = 10.0) -> bool:

    # Read the frames stored
    frame_A = cv2.imread(frame_path_A)
    frame_B = cv2.imread(frame_path_B)

    # Check if frame A is present in the drive
    if frame_A is None:
        raise FileNotFoundError(f"Could not load frame: {frame_path_A}")

    # Check if frame B is present in the drive
    if frame_B is None:
        raise FileNotFoundError(f"Could not load frame: {frame_path_B}")

    # Convert the frames into grayscale for comparison
    frame_A_gray = cv2.cvtColor(frame_A, cv2.COLOR_BGR2GRAY)
    frame_B_gray = cv2.cvtColor(frame_B, cv2.COLOR_BGR2GRAY)

    # Use absolute difference to check the difference between the 2 frames
    difference = cv2.absdiff(frame_A_gray, frame_B_gray)

    # Calculate the mean difference
    mean_diff = np.mean(difference)

    # Check if it is duplicated via the threshold
    if mean_diff < threshold:
        return True

    # Otherwise, the frames are different enough
    return False


def filter_unique_frames(all_frames: list[dict], threshold: float = 10.0) -> list[dict]:

    # Create an empty list to store only the unique frames
    unique_frames = []

    # Loop through each frame dictionary in the full frame list
    for frame in all_frames:

        # If no frames have been kept yet, always keep the first frame
        if not unique_frames:
            unique_frames.append(frame)
            continue

        # Get the file path of the current frame being checked
        current_frame_path = frame["path"]

        # Get the file path of the last unique frame that was kept
        last_unique_frame_path = unique_frames[-1]["path"]

        # Compare the current frame against the last kept unique frame
        duplicate = is_duplicate(
            current_frame_path,
            last_unique_frame_path,
            threshold=threshold,
        )

        # If the current frame is not a duplicate, keep it
        if not duplicate:
            unique_frames.append(frame)

        # If it is a duplicate, skip it
        else:
            print(f"[DEDUP] Skipped duplicate frame: {frame['frame_index']}")

    # Return the filtered list containing only unique frames
    return unique_frames