from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

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

        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            _attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16
        )

        if not torch.cuda.is_available():
            raise RuntimeError("DeepSeek-OCR with FlashAttention 2 requires a CUDA GPU runtime.")

        model = model.to("cuda").eval()

        print("[INFO] DeepSeek-OCR model loaded successfully.")

    return tokenizer, model


def extract_text_deepseek(image_path: str):
    """
    Run DeepSeek-OCR on a single image and return output
    in a structure compatible with the existing pipeline.
    """
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

    extracted_text = str(response).strip()

    return [
        {
            "bbox": None,
            "text": extracted_text,
            "confidence": 1.0,
            "is_low_confidence": False
        }
    ]