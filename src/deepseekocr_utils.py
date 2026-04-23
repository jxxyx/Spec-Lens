from pathlib import Path
import io
from contextlib import redirect_stdout

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
    """

    def __init__(self):
        self._tokenizer = None
        self._model = None

    def download(self) -> None:
        """
        Download model weights to the HuggingFace cache and apply the source
        patch to the cached files. Call this ONCE in a setup cell and then
        restart the Colab runtime before running inference.
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
        Import and load the model into memory. Only called on first inference.
        Assumes download() (and a runtime restart) has already been run.
        """
        if self._model is not None:
            return

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

    def _clean_captured_stdout(self, raw_stdout: str) -> str:
        """
        Extract OCR text from DeepSeek's printed console output.

        DeepSeek-OCR often prints the recognized text to stdout and returns None.
        This helper removes diagnostic/log lines and keeps the actual OCR text.
        """
        if not raw_stdout:
            return ""

        keep_lines = []
        for line in raw_stdout.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            # Skip separator lines
            if stripped.startswith("==="):
                continue

            # Skip known diagnostic / metadata lines
            if stripped.startswith("BASE:"):
                continue
            if stripped.startswith("PATCHES:"):
                continue
            if stripped.startswith("image size:"):
                continue
            if stripped.startswith("valid image tokens:"):
                continue
            if stripped.startswith("output texts tokens"):
                continue
            if stripped.startswith("compression ratio:"):
                continue
            if stripped.startswith("image:"):
                continue
            if stripped.startswith("other:"):
                continue
            if stripped.startswith("[INFO]"):
                continue
            if stripped.startswith("[WARNING]"):
                continue

            keep_lines.append(stripped)

        return "\n".join(keep_lines).strip()

    def extract_text(self, image_path: str) -> list[dict]:
        """
        Extract text from an image using DeepSeek-OCR.

        Returns:
            list[dict]: Each dict has bbox, text, confidence, is_low_confidence.
                        bbox is None because DeepSeek returns full-page text.
                        Returns an empty list only if no usable OCR text is found.
        """
        self._load()

        Path(_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        buffer = io.StringIO()

        with torch.no_grad():
            with redirect_stdout(buffer):
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

        captured_stdout = buffer.getvalue()
        captured_text = self._clean_captured_stdout(captured_stdout)

        # Prefer direct model return if meaningful
        if response is not None:
            text = str(response).strip()
            if text and text.lower() != "none":
                return [{
                    "bbox": None,
                    "text": text,
                    "confidence": 1.0,
                    "is_low_confidence": False,
                }]

        # Fallback to OCR text printed to stdout
        if captured_text:
            return [{
                "bbox": None,
                "text": captured_text,
                "confidence": 1.0,
                "is_low_confidence": False,
            }]

        print(f"[WARNING] DeepSeek produced no usable OCR text for {image_path}.")
        return []


# Module-level singleton — shared across the pipeline so weights load once only.
engine = DeepSeekOCREngine()


def extract_text_deepseek(image_path: str) -> list[dict]:
    """
    Public entry point matching the EasyOCR interface in ocr_utils.py.
    """
    return engine.extract_text(image_path)