import easyocr

_reader = None


def _get_reader():
    """Lazy-load the EasyOCR reader so it only initialises when first needed."""
    global _reader
    if _reader is None:
        print("[INFO] Loading EasyOCR model...")
        _reader = easyocr.Reader(['en'])
        print("[INFO] EasyOCR model loaded.")
    return _reader


def extract_text_from_image(image_path: str, min_confidence: float = 0.50):
    """
    Extract text from an image using EasyOCR.

    Args:
        image_path (str): Path to the image file.
        min_confidence (float): Minimum confidence threshold for flagging low-quality results.

    Returns:
        list[dict]: Each dict contains bbox, text, confidence, and is_low_confidence flag.
    """
    reader = _get_reader()
    results = reader.readtext(image_path)

    extracted = []
    for bbox, text, confidence in results:
        bbox_clean = [[int(p[0]), int(p[1])] for p in bbox]
        extracted.append({
            "bbox": bbox_clean,
            "text": text,
            "confidence": round(float(confidence), 4),
            "is_low_confidence": float(confidence) < min_confidence
        })

    return extracted