#This is to read the text that are extracted from the image from video_utils.py

#Import the library
import easyocr

reader = easyocr.Reader(['en'])

def extract_text_from_image(image_path: str):
    #Reads the image to find text
    results = reader.readtext(image_path)
    return [item[1] for item in results]