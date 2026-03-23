#To run the code to test if the skeleton works

#Import function from src
from src.pipeline import process_video

if __name__ == "__main__":
    video_path = "data/raw/video1.mp4"

    results = process_video(video_path)

    for r in results:
        #Print the file name so you can verify the image exists
        print("Frame:", r["frame"])
        #Print the list of strings found on that specific screen
        print("Text:", r["text"])
        #A visual separator to make the terminal output easier to read
        print("-" * 30)