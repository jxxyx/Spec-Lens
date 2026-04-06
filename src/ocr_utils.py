#This is to read the text that are extracted from the image from video_utils.py

#Import the library
import easyocr

reader = easyocr.Reader(['en'])

def extract_text_from_image(image_path: str, min_confidence: float = 0.50):
    #Reads the image to find text
    results = reader.readtext(image_path)

    extracted = []
    for item in results:
        bbox, text, confidence = item
        bbox_clean = [[int(point[0]), int(point[1])] for point in bbox]
        extracted.append({
            "bbox": bbox_clean,
            "text": text,
            "confidence": float(confidence),
            "is_low_confidence": float(confidence) < min_confidence
        })

    return extracted

    