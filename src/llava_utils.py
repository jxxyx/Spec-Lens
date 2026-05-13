from pathlib import Path
import torch
import re
from PIL import Image

MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"

# ── Prompt template ────────────────────────────────────────────────────────────
# The image is passed directly to LLaVA alongside this text prompt.
# cleaned_text (OCR output) is injected as additional grounding context so the
# model can reconcile what it sees with what OCR already extracted.
_PROMPT_TEMPLATE = """You are a Senior Business Analyst documenting ONE mobile banking UI screen.

OCR text extracted from this screen:
{ocr_text}

Your job is to produce structured Business Analysis (BA) documentation for each distinct business capability or user goal that is directly visible on this screen.

Use the ROCSTAR framework for each user story:
- Role
- Objective
- Context
- Scenario
- Trigger
- Action
- Result

STRICT RULES:
- Only document what is explicitly visible in the screenshot and OCR text.
- Do NOT invent hidden functionality, backend logic, future screens, workflows, validations, or UI elements that are not shown.
- Do NOT mention missing features unless the visible screen creates a clear ambiguity.
- Do NOT speculate about downstream screens, backend validations, payee management, authentication, transaction filtering, reporting, downloads, sorting, searching, or transaction details unless those elements are visibly shown.
- Generate one user story for each distinct visible business capability or user goal.
- Group related UI elements under the same user story when they support the same objective.
- Identify as many user stories as the screen warrants, based solely on visible evidence.
- If behaviour is inferred but not directly visible, record it under "Assumptions" and assign a Medium or Low confidence level.
- Every user story must include visible Evidence.
- Every statement must be traceable to visible evidence in the screenshot or OCR text.
- Clearly separate confirmed observations from assumptions.
- Use concise, factual language.
- If a category has no grounded findings, write: "None identified from this screen."

IMPORTANT OUTPUT RULES:
- Describe only what can be observed on this screen.
- Do NOT describe what happens after a user taps a button unless the next screen is shown.
- For example, write:
  "Then the Transfer option is available for selection."
  NOT:
  "Then the user is prompted to enter transfer details."

OUTPUT FORMAT (follow exactly):

SCREEN SUMMARY
[2–3 sentences based only on what is visible on this screen. Describe:
- the likely screen name,
- the intended user,
- and the primary purpose of this screen.
Do not infer app-wide context or downstream processes.]

USER STORIES

US-1: [Short Title]

Priority: [Primary | Secondary | Informational]
Confidence: [High | Medium | Low]

Role: [User role]
Objective: [What the user wants to achieve]
Context: [Where this occurs in the visible user journey]
Result: [Expected business outcome visible or reasonably implied]

User Story:
As a [Role],
I want to [Objective],
In the context of [Context],
So that [Result].

Trigger: [What causes the user to perform this action]
Action: [What the user physically does on this screen]

Evidence:
- [Visible UI elements, labels, buttons, values, or OCR text that support this story]

Assumptions:
- [Any inferred behaviour not directly visible]
- [Write "None" if no assumptions were made]

Scenario: [Scenario Name]
Given [precondition visible or reasonably implied]
When [trigger/action]
Then [expected visible result on this screen]

US-2: [Short Title]

[Repeat the exact same structure]

[Continue for all distinct user goals visible on the screen]

ACCEPTANCE CRITERIA

Scenario: [Scenario Name from US-1]
Given ...
When ...
Then ...

Scenario: [Scenario Name from US-2]
Given ...
When ...
Then ...

[Continue for all scenarios from all user stories]

GAP ANALYSIS

Missing Behaviour
- Gap: [Visible UI element whose behaviour is unclear, or "None identified from this screen."]
  Evidence: [Visible UI element or OCR text]

Validation Rules Not Defined
- Gap: [Only if a visible input field or form requires validation; otherwise "None identified from this screen."]
  Evidence: [Visible input field or action]

Assumptions Requiring Confirmation
- Gap: [Only assumptions explicitly listed in the user stories above, or "None identified from this screen."]
  Evidence: [Related user story or visible UI element]

Ambiguous Labels or Icons
- Gap: [Only labels or icons whose meaning is unclear, or "None identified from this screen."]
  Evidence: [Visible label or icon]

Accessibility Concerns
- Gap: [Only observable accessibility concerns, or "None identified from this screen."]
  Evidence: [Visible UI/text/icon]

Implied Business Rules
- Gap: [Only business rules directly suggested by visible values or text, or "None identified from this screen."]
  Evidence: [Visible value, label, or OCR text]
"""

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
        max_new_tokens: int = 1024,
    ) -> dict:
        """
        Run LLaVA over one frame and return structured BA artefacts.

        Args:
            image_path (str): Path to the frame JPEG.
            cleaned_text (list[str]): OCR-cleaned text strings.
            max_new_tokens (int): Token budget for the generated response.
                                  1024 is required for the full ROCSTAR output
                                  (6+ user stories + AC + GAP can exceed 900 tokens).

        Returns:
            dict with keys:
                "screen_summary"        — str
                "user_stories"          — list[dict] with title, story, scenario
                "acceptance_criteria"   — str (consolidated Gherkin block)
                "gap_analysis"          — str (categorised bullet list)
                "truncated"             — bool (True if response was cut off)
                "raw_response"          — str
                "image_path"            — str
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
    Parse LLaVA's ROCSTAR-structured output into typed fields.

    Top-level sections detected (case-insensitive, optional bold markdown):
        SCREEN SUMMARY
        USER STORIES
        ACCEPTANCE CRITERIA
        GAP ANALYSIS

    Within USER STORIES, each US-N block is parsed into:
        title     — short title from the US-N: header
        story     — ROCSTAR fields (Role, Objective, ... Action, Evidence, Assumptions)
        scenario  — Gherkin block (Given/When/Then)

    Truncation is flagged when both acceptance_criteria and gap_analysis are
    empty — indicating the model ran out of token budget before finishing.

    Falls back gracefully if headers are missing or malformed.
    """
    result = {
        "screen_summary": "",
        "user_stories": [],
        "acceptance_criteria": "",
        "gap_analysis": "",
        "truncated": False,
    }

    # ── 1. Split into top-level sections ──────────────────────────────────────
    section_pattern = re.compile(
        r"\*{0,2}\s*(SCREEN SUMMARY|USER STORIES|ACCEPTANCE CRITERIA|GAP ANALYSIS)\s*\*{0,2}",
        re.IGNORECASE,
    )

    matches = list(section_pattern.finditer(text))

    if not matches:
        # No structure found — store everything as screen_summary fallback
        result["screen_summary"] = text.strip()
        result["truncated"] = True
        return result

    sections = {}
    for i, m in enumerate(matches):
        key = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[key] = text[start:end].strip()

    result["screen_summary"]      = sections.get("SCREEN SUMMARY", "")
    result["acceptance_criteria"] = sections.get("ACCEPTANCE CRITERIA", "")
    result["gap_analysis"]        = sections.get("GAP ANALYSIS", "")

    # ── 2. Parse individual user stories ──────────────────────────────────────
    us_block = sections.get("USER STORIES", "")

    if us_block:
        # Split on "US-N: Title" headers (optional bold, flexible spacing)
        us_parts = re.split(
            r"\*{0,2}\s*US-\d+\s*:\s*([^\n*]+)\*{0,2}",
            us_block,
        )
        # Pattern: [pre-text, title1, body1, title2, body2, ...]
        i = 1
        while i < len(us_parts) - 1:
            title = us_parts[i].strip()
            body  = us_parts[i + 1].strip() if i + 1 < len(us_parts) else ""

            # Split body at the Scenario block
            scenario_match = re.search(
                r"(Scenario\s*:.*)",
                body,
                re.IGNORECASE | re.DOTALL,
            )

            if scenario_match:
                story_text    = body[: scenario_match.start()].strip()
                scenario_text = scenario_match.group(1).strip()
            else:
                story_text    = body.strip()
                scenario_text = ""

            result["user_stories"].append({
                "title":    title,
                "story":    story_text,
                "scenario": scenario_text,
            })

            i += 2

    # ── 3. Truncation detection ────────────────────────────────────────────────
    # If both AC and GAP are empty the model almost certainly ran out of tokens.
    if not result["acceptance_criteria"] and not result["gap_analysis"]:
        result["truncated"] = True

    return result


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