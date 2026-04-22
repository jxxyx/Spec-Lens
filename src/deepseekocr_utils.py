from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from src.utils.deepseek_patch import patch_deepseek

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
_OUTPUT_DIR = "/content/deepseek_outputs"


class DeepSeekOCREngine:
    """
    Wrapper around the DeepSeek-OCR model.

    Loading is split into two explicit steps to avoid the patch-after-import race:

        Step 1 — download():
            Uses snapshot_download() to pull the model files to the HuggingFace
            cache WITHOUT importing or executing any remote code.
            patch_deepseek() is then applied to the cached files on disk.

        Step 2 — _load() (called automatically on first inference):
            Calls AutoModel.from_pretrained() with trust_remote_code=True.
            Because the files are already patched on disk, the remote code that
            gets imported and compiled is the patched version from the start.
            There is no window where unpatched code runs in-memory.

    Intended Colab setup cell:
        from src.deepseekocr_utils import engine as deepseek_engine
        deepseek_engine.download()   # download + patch, then restart runtime
        # After restart, inference calls _load() automatically on first use.
    """

    def __init__(self):
        self._tokenizer = None
        self._model = None

    def download(self) -> None:
        """
        Download model weights to the HuggingFace cache and apply the source
        patch to the cached files.  Call this ONCE in a setup cell and then
        restart the Colab runtime before running inference.

        This ensures that when trust_remote_code=True imports the remote
        modeling file, it loads the already-patched version.
        """
        print("[INFO] Downloading DeepSeek-OCR model files (no import yet)...")
        snapshot_path = snapshot_download(MODEL_NAME)
        print(f"[INFO] Download complete at: {snapshot_path}")
        print("[INFO] Applying patch to downloaded files...")
        patch_deepseek(base_path=snapshot_path)
        print(
            "[INFO] Patch applied.\n"
            "[ACTION] Restart the Colab runtime now, then run your inference cell."
        )

    def _load(self) -> None:
        """
        Import and load the model into memory.  Only called on first inference.
        Assumes download() (and a runtime restart) has already been run so the
        patched remote code is what gets compiled by trust_remote_code=True.
        """
        if self._model is not None:
            return

        # Import here (not at module top) so that on a fresh runtime the remote
        # modeling code is not imported until after the patch has been applied.
        from transformers import AutoModel, AutoTokenizer

        print("[INFO] Loading DeepSeek-OCR model into memory...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self._model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        )

        if torch.cuda.is_available():
            self._model = self._model.to("cuda")

        self._model = self._model.eval()
        print("[INFO] DeepSeek-OCR model ready.")

    def extract_text(self, image_path: str) -> list[dict]:
        """
        Extract text from an image using DeepSeek-OCR.

        Args:
            image_path (str): Path to the image file.

        Returns:
            list[dict]: Each dict has bbox, text, confidence, is_low_confidence.
                        bbox is None — DeepSeek returns full-page text, not bounding boxes.
                        Returns an empty list if the model produces no output.
        """
        self._load()

        Path(_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            response = self._model.infer(
                self._tokenizer,
                prompt="<image>\nFree OCR.",
                image_file=str(Path(image_path)),
                output_path=_OUTPUT_DIR,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
                test_compress=True,
            )

        # model.infer() can return None if it produces no output (e.g. blank image,
        # inference error swallowed internally).  Guard against this explicitly
        # rather than converting None to the string "None".
        if response is None:
            print(f"[WARNING] DeepSeek returned None for {image_path}. Skipping.")
            return []

        text = str(response).strip()
        if not text:
            print(f"[WARNING] DeepSeek returned empty string for {image_path}. Skipping.")
            return []

        return [{
            "bbox": None,
            "text": text,
            "confidence": 1.0,
            "is_low_confidence": False,
        }]


# Module-level singleton — shared across the pipeline so weights load once only.
engine = DeepSeekOCREngine()


def extract_text_deepseek(image_path: str) -> list[dict]:
    """
    Public entry point matching the EasyOCR interface in ocr_utils.py.

    Args:
        image_path (str): Path to the image file.

    Returns:
        list[dict]: OCR results. Empty list if the model produced no output.
    """
    return engine.extract_text(image_path)