#Core of the whole application

#Import the other py files to read their functions
from src.video_utils import extract_frames
from src.ocr_utils import extract_text_from_image

#function to process the video
def process_video(video_path: str):
    #This returns a list of file paths
    frames = extract_frames(video_path, "data/frames")

    #INITIALIZE: Create an empty list to store our final combined results
    all_text = []

    #Loop through every single image we just created
    for frame in frames:
        #Pass the current image path to the OCR function to get the text inside it
        text = extract_text_from_image(frame)

        #Pair the image path with its text so we know exactly WHERE the words came from
        all_text.append({
            "frame": frame,
            "text": text
        })

    return all_text