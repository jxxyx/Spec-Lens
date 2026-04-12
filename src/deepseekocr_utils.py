from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
from src.utils.deepseek_patch import patch_deepseek

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

tokenizer = None
model = None


def load_deepseek_model():
    global tokenizer, model

    if tokenizer is None or model is None:
        print("[INFO] Loading DeepSeek-OCR model...")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=dtype,
            attn_implementation="eager"
        )

        if torch.cuda.is_available():
            model = model.to("cuda")

        model = model.eval()

        print("[INFO] DeepSeek-OCR model loaded successfully.")

        patch_deepseek()

    return tokenizer, model


def extract_text_deepseek(image_path: str):
    tokenizer, model = load_deepseek_model()

    image_file = str(Path(image_path))
    output_dir = "/content/deepseek_outputs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    prompt = "<image>\nFree OCR."

    with torch.no_grad():
        response = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_file,
            output_path=output_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True,
            test_compress=True
        )

    return [{
        "bbox": None,
        "text": str(response).strip(),
        "confidence": 1.0,
        "is_low_confidence": False
    }]