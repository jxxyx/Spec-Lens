from pathlib import Path
import re

# Sentinel string written into the file after a successful patch.
# Used to detect files that are already patched so we can skip them safely.
_PATCH_SENTINEL = "# [deepseek-patch-applied]"

_PATTERN = r"""
    (?P<indent>\s*)inputs_embeds\[idx\]\.masked_scatter_\(
    \s*images_seq_mask\[idx\]\.unsqueeze\(-1\).*?,
    \s*images_in_this_batch
    \s*\)
"""

_REPLACEMENT = (
    r"\g<indent># [deepseek-patch-applied]\n"
    r"\g<indent>mask = images_seq_mask[idx].unsqueeze(-1)"
    r".to(device=inputs_embeds.device).bool()\n"
    r"\g<indent>source = images_in_this_batch"
    r".to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)\n"
    r"\g<indent>inputs_embeds[idx].masked_scatter_(mask, source)"
)


def patch_deepseek() -> None:
    """
    Patch the cached DeepSeek-OCR modeling file to fix a device/dtype mismatch
    in the upstream masked_scatter_ call.

    The patch injects explicit .to(device=...) and .to(dtype=...) calls so that
    the mask and source tensors are on the same device as inputs_embeds before
    the scatter operation runs.  Without this, CUDA float16 inference raises a
    RuntimeError on a fresh HuggingFace cache.

    This function is idempotent — calling it multiple times on an already-patched
    file is safe and produces no changes.

    Must be called AFTER snapshot_download() (so the files exist on disk) but
    BEFORE AutoModel.from_pretrained() with trust_remote_code=True (so the
    imported remote code is the patched version).
    """
    base = Path(
        "/root/.cache/huggingface/modules/transformers_modules"
        "/deepseek-ai/DeepSeek-OCR"
    )
    targets = list(base.rglob("modeling_deepseekocr.py"))

    if not targets:
        print(
            "[WARNING] patch_deepseek: no modeling_deepseekocr.py found under\n"
            f"          {base}\n"
            "          Has snapshot_download() been called yet?"
        )
        return

    for path in targets:
        text = path.read_text()

        # Idempotency check — skip files that carry the sentinel from a prior run
        if _PATCH_SENTINEL in text:
            print(f"[PATCH] Already patched, skipping: {path}")
            continue

        new_text, count = re.subn(_PATTERN, _REPLACEMENT, text, flags=re.VERBOSE)

        if count > 0:
            path.write_text(new_text)
            print(f"[PATCH] Patched successfully: {path}")
        else:
            print(
                f"[PATCH] Pattern not found in {path}.\n"
                "        The upstream model code may have changed — "
                "review masked_scatter_ usage manually."
            )