from pathlib import Path
import torch
import re
from PIL import Image

MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"

# ── Prompt template ────────────────────────────────────────────────────────────
# The image is passed directly to LLaVA alongside this text prompt.
# cleaned_text (OCR output) is injected as additional grounding context so the
# model can reconcile what it sees with what OCR already extracted.
_PROMPT_TEMPLATE = """You are an expert Business Analyst reviewing a UI screenshot.

The following text was extracted from this screen via OCR:
{ocr_text}

Based on the screenshot and the extracted text above, generate the following:

1. USER STORY
   Format: As a [user], I want to [action], so that [benefit].

2. ACCEPTANCE CRITERIA (Gherkin)
   Format:
   Given [precondition]
   When [action]
   Then [expected result]
   (Add more Given/When/Then steps if needed)

3. GAP ANALYSIS
   List any UI elements, validation rules, edge cases, or system behaviours
   that are visible or implied but NOT captured in the user story above.

Be specific to what is visible in the screenshot. Do not invent fields or
behaviours that are not present or reasonably implied."""


class LLaVAEngine:
    """
    Wrapper around LLaVA-1.6-Mistral-7B for multimodal BA artefact generation.

    Uses 4-bit BitsAndBytes quantisation so the model fits within the 15GB
    VRAM of a Colab T4 GPU. The model is lazy-loaded on first inference call.

    VRAM note: Do NOT have DeepSeek-OCR loaded at the same time.
    Free it first with:
        import torch, gc
        del model; gc.collect(); torch.cuda.empty_cache()
    """

    def __init__(self):
        self._model = None
        self._processor = None

    def _load(self) -> None:
        """Load model and processor into memory on first use."""
        if self._model is not None:
            return

        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
            BitsAndBytesConfig,
        )

        print("[INFO] Loading LLaVA-1.6-Mistral-7B (4-bit quantised)...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self._processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)

        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",  # model split across devices
            torch_dtype=torch.float16,
        )

        self._model.eval()
        print("[INFO] LLaVA-1.6 ready.")

    def analyse(
        self,
        image_path: str,
        cleaned_text: list[str],
        max_new_tokens: int = 512,
    ) -> dict:
        """
        Run LLaVA over one frame and return structured BA artefacts.

        Args:
            image_path (str): Path to the frame JPEG.
            cleaned_text (list[str]): OCR-cleaned text strings.
            max_new_tokens (int): Token budget for the generated response.

        Returns:
            dict with keys:
                "user_story"
                "acceptance_criteria"
                "gap_analysis"
                "raw_response"
                "image_path"
        """
        self._load()

        image = Image.open(image_path).convert("RGB")
        ocr_text = "\n".join(cleaned_text) if cleaned_text else "(no OCR text extracted)"
        prompt = _PROMPT_TEMPLATE.format(ocr_text=ocr_text)

        # LLaVA-Next conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        formatted_prompt = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self._processor(
            images=image,
            text=formatted_prompt,
            return_tensors="pt",
        )

        # ✅ FIX: Safe device handling for device_map="auto"
        if torch.cuda.is_available():
            inputs = {
                k: v.to("cuda") if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic output (important for evaluation)
            )

        # Decode only generated tokens
        input_len = inputs["input_ids"].shape[1]

        raw_response = self._processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()

        parsed = _parse_response(raw_response)
        parsed["raw_response"] = raw_response
        parsed["image_path"] = image_path

        return parsed

    def unload(self) -> None:
        """
        Release model weights from GPU memory.

        Call this before loading DeepSeek-OCR in the same session.
        """
        import gc

        self._model = None
        self._processor = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[INFO] LLaVA model unloaded and VRAM freed.")


def _parse_response(text: str) -> dict:
    """
    Parse the model's free-text output into structured fields.

    Looks for numbered section headers:
    1. USER STORY
    2. ACCEPTANCE CRITERIA
    3. GAP ANALYSIS
    """
    sections = {
        "user_story": "",
        "acceptance_criteria": "",
        "gap_analysis": "",
    }

    parts = re.split(
        r"\n?\s*\d+\.\s+(?:USER STORY|ACCEPTANCE CRITERIA|GAP ANALYSIS)\s*\n",
        text,
        flags=re.IGNORECASE,
    )

    if len(parts) >= 2:
        sections["user_story"] = parts[1].strip()

    if len(parts) >= 3:
        sections["acceptance_criteria"] = parts[2].strip()

    if len(parts) >= 4:
        sections["gap_analysis"] = parts[3].strip()

    # Fallback: keep full output
    if not any(sections.values()):
        sections["user_story"] = text.strip()

    return sections


# Singleton instance
engine = LLaVAEngine()


def analyse_frame(image_path: str, cleaned_text: list[str]) -> dict:
    """
    Public entry point for reasoning layer.

    Args:
        image_path (str)
        cleaned_text (list[str])

    Returns:
        dict of structured BA artefacts
    """
    return engine.analyse(image_path, cleaned_text)