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


def patch_deepseek(base_path: str | None = None) -> None:
    """
    Patch the DeepSeek-OCR modeling file to fix a device/dtype mismatch
    in the upstream masked_scatter_ call.

    The patch injects explicit .to(device=...) and .to(dtype=...) calls so that
    the mask and source tensors are on the same device/dtype as inputs_embeds
    before the scatter operation runs.

    Args:
        base_path (str | None):
            Root folder to search for modeling_deepseekocr.py.
            - If provided, search there first (recommended: pass snapshot_download() path)
            - If None, fall back to the transformers_modules cache path

    This function is idempotent — calling it multiple times on an already-patched
    file is safe and produces no changes.
    """

    search_roots = []

    if base_path is not None:
        search_roots.append(Path(base_path))

    # Fallback path used by trust_remote_code imports
    search_roots.append(
        Path("/root/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR")
    )

    targets = []
    searched = []

    for root in search_roots:
        searched.append(str(root))
        if root.exists():
            found = list(root.rglob("modeling_deepseekocr.py"))
            if found:
                targets.extend(found)

    # Remove duplicates while preserving order
    unique_targets = []
    seen = set()
    for path in targets:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_targets.append(path)

    if not unique_targets:
        print(
            "[WARNING] patch_deepseek: no modeling_deepseekocr.py found under:\n"
            + "\n".join(f"  - {p}" for p in searched)
        )
        return

    for path in unique_targets:
        text = path.read_text(encoding="utf-8")

        # Idempotency check — skip files that carry the sentinel from a prior run
        if _PATCH_SENTINEL in text:
            print(f"[PATCH] Already patched, skipping: {path}")
            continue

        new_text, count = re.subn(_PATTERN, _REPLACEMENT, text, flags=re.VERBOSE)

        if count > 0:
            path.write_text(new_text, encoding="utf-8")
            print(f"[PATCH] Patched successfully: {path}")
        else:
            print(
                f"[PATCH] Pattern not found in {path}.\n"
                "        The upstream model code may have changed — "
                "review masked_scatter_ usage manually."
            )