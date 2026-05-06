from pathlib import Path
import torch
import re
from PIL import Image

MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"

# ── Prompt template ────────────────────────────────────────────────────────────
# The image is passed directly to LLaVA alongside this text prompt.
# cleaned_text (OCR output) is injected as additional grounding context so the
# model can reconcile what it sees with what OCR already extracted.
_PROMPT_TEMPLATE = """You are a Business Analyst writing documentation for ONE specific screen.

STRICT RULES — you must follow these exactly:
- Only describe UI elements and actions that are LITERALLY VISIBLE in this screenshot.
- Do NOT invent features, buttons, or behaviours that are not shown on screen.
- Do NOT write generic banking requirements. Be specific to THIS screen only.
- Keep Acceptance Criteria to 3-5 steps maximum, covering only what is visible.
- Each section must end before the next numbered heading begins.

OCR text extracted from this screen:
{ocr_text}

Now produce exactly three sections using these headings:

1. USER STORY
As a [specific user type visible on screen], I want to [specific action shown on screen], so that [direct benefit of this screen].

2. ACCEPTANCE CRITERIA
Given [the specific screen state shown]
When [a specific action on a visible element]
Then [the direct visible outcome]
(Maximum 5 Given/When/Then blocks. Only cover what is on screen.)

3. GAP ANALYSIS
List only: (a) visible UI elements not covered by the user story, (b) missing validation that a BA would flag for THIS screen, (c) unclear or ambiguous elements actually visible. Maximum 5 bullet points. Do not pad with generic banking features."""


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
    Parse LLaVA's free-text output into structured BA fields.

    Handles:
    - Numbered headers:  1. USER STORY
    - Bold markdown:     **1. USER STORY**
    - Missing sections:  gracefully returns empty string
    - Fallback:          full text goes into user_story if no headers found
    """
    sections = {
        "user_story": "",
        "acceptance_criteria": "",
        "gap_analysis": "",
    }

    # Pattern matches: optional **, digit, dot, section name, optional **
    # e.g. "1. USER STORY", "**2. ACCEPTANCE CRITERIA**", "3. Gap Analysis"
    header_pattern = re.compile(
        r"\*{0,2}\s*\d+\.\s*"
        r"(USER STORY|ACCEPTANCE CRITERIA(?:\s*\(Gherkin\))?|GAP ANALYSIS)"
        r"\s*\*{0,2}",
        re.IGNORECASE,
    )

    # Find all section headers and their positions
    matches = list(header_pattern.finditer(text))

    if not matches:
        # No structured headers found — put everything in user_story as fallback
        sections["user_story"] = text.strip()
        return sections

    for i, match in enumerate(matches):
        section_name = match.group(1).upper().split("(")[0].strip()  # normalise
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if "USER STORY" in section_name:
            sections["user_story"] = content
        elif "ACCEPTANCE CRITERIA" in section_name:
            sections["acceptance_criteria"] = content
        elif "GAP ANALYSIS" in section_name:
            sections["gap_analysis"] = content

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