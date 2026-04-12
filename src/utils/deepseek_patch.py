from pathlib import Path
import re

def patch_deepseek():
    base = Path("/root/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR")

    targets = list(base.rglob("modeling_deepseekocr.py"))

    if not targets:
        print("[WARNING] DeepSeek files not found. Did you load the model first?")
        return

    for path in targets:
        text = path.read_text()

        pattern = r"""
        (?P<indent>\s*)inputs_embeds\[idx\]\.masked_scatter_\(
        \s*images_seq_mask\[idx\]\.unsqueeze\(-1\).*?,
        \s*images_in_this_batch
        \s*\)
        """

        replacement = r"""\g<indent>mask = images_seq_mask[idx].unsqueeze(-1).to(device=inputs_embeds.device).bool()
\g<indent>source = images_in_this_batch.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
\g<indent>inputs_embeds[idx].masked_scatter_(mask, source)"""

        new_text, count = re.subn(pattern, replacement, text, flags=re.VERBOSE)

        if count > 0:
            path.write_text(new_text)
            print(f"[PATCHED] {path}")
        else:
            print(f"[SKIPPED] {path}")