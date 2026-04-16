import re

# Tokens that are short but semantically meaningful in UI / banking contexts
_ALLOWLIST = {
    "id", "ok", "no", "go", "ui", "db", "fx", "atm", "pin", "otp",
    "ref", "sgd", "usd", "uob", "mfa", "sso", "api", "url", "pdf",
}


def _is_likely_noise(text: str) -> bool:
    """
    Return True if a text fragment looks like OCR noise rather than real content.
    Short strings are noise unless they appear on the allowlist or look like
    meaningful codes (all-digits, currency amounts, version strings, etc.).
    """
    # Allow anything on the explicit allowlist (case-insensitive)
    if text.lower() in _ALLOWLIST:
        return False

    # Allow short numeric / currency strings: "42", "$5", "v2", "£10"
    if re.fullmatch(r"[$£€¥]?\d+(\.\d+)?|v\d+(\.\d+)*", text, re.IGNORECASE):
        return False

    # Drop anything 2 characters or shorter that didn't match above
    if len(text) <= 2:
        return True

    # Drop strings that are pure punctuation or whitespace
    if re.fullmatch(r"[\W_]+", text):
        return True

    return False


def clean_ocr_results(
    ocr_results: list[dict],
    min_confidence: float = 0.5,
) -> list[str]:
    """
    Filter and clean raw OCR output into a list of usable text strings.

    Args:
        ocr_results (list[dict]): Raw output from extract_text_from_image()
                                  or extract_text_deepseek().
        min_confidence (float):   Drop results below this confidence score.

    Returns:
        list[str]: Cleaned text fragments, in original detection order.
    """
    cleaned = []

    for item in ocr_results:
        text = item["text"].strip()
        confidence = item["confidence"]

        if confidence < min_confidence:
            continue

        if _is_likely_noise(text):
            continue

        cleaned.append(text)

    return cleaned