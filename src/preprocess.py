def clean_ocr_results(ocr_results, min_confidence=0.5):
    cleaned = []

    for item in ocr_results:
        text = item["text"].strip()
        confidence = item["confidence"]

        # Remove low confidence
        if confidence < min_confidence:
            continue

        # Remove very short garbage (like 't', '6')
        if len(text) <= 2:
            continue

        cleaned.append(text)

    return cleaned