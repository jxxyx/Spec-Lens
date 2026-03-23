#This is the script that will be used to handle video input and transforming it into frames

#Import the necessary library
import cv2
from pathlib import Path

def extract_frames(video_path: str, output_folder: str, interval: int = 30):
    #Convert folder string into a Path Object and create folder if the folder does not exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    #Start the process by opening the video file to read
    cap = cv2.VideoCapture(video_path)

    #Counts total frame in the video
    frame_count = 0 
    #Counts frames that has been saved as images
    saved_count = 0
    saved_frames = []

    #Function to read the video frame by frame
    while True:
        #frame is the actual image data for the specific moment
        success, frame = cap.read()

        #Break the process if the video has reached the end
        if not success:
            break
        
        #This is the function to save every X frames (the interval)
        if frame_count % interval == 0:
            #Create a filename like "data/frames/frame_0.jpg"
            frame_file = output_path / f"frame_{saved_count}.jpg"

            #Write the image data to a physical file on your hard drive
            cv2.imwrite(str(frame_file), frame)

            #Record the path so our pipeline knows where to find it later
            saved_frames.append(str(frame_file))
            saved_count += 1

        frame_count += 1

    #Close video file to free up system memory
    cap.release()
    return saved_frames