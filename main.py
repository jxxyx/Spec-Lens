# from src.pipeline import process_video
# import json

# if __name__ == "__main__":
#     video_path = "data/raw/video1.mp4"

#     # Google Drive base folder for persistent outputs in Colab
#     drive_base = "/content/drive/MyDrive/Spec-Lens-data"

#     # ------------------------------------------
#     # RESUME TEST SETTINGS
#     # ------------------------------------------
#     # FIRST TEST RUN:
#     # clear_frames = True
#     # resume = True
#     # max_frames = 3
#     #
#     # SECOND TEST RUN:
#     # clear_frames = False
#     # resume = True
#     # max_frames = None
#     # ------------------------------------------

#     clear_frames = False
#     resume = True
#     max_frames = None

#     results = process_video(
#         video_path=video_path,
#         output_folder=f"{drive_base}/frames",
#         checkpoint_base_folder=f"{drive_base}/ocr_results",
#         interval=30,
#         clear_frames=clear_frames,
#         ocr_engine="easyocr",
#         resume=resume,
#         max_frames=max_frames
#     )

#     for frame_result in results:
#         print(f"\nFrame: {frame_result['frame']}")

#         print("\nRAW OCR:")
#         for item in frame_result["ocr_results"]:
#             flag = "LOW CONFIDENCE" if item["is_low_confidence"] else "OK"
#             print(f"[{flag}] {item['text']} ({item['confidence']:.2f})")

#         print("\nCLEANED:")
#         for text in frame_result["cleaned_text"]:
#             print(text)

#     with open(f"{drive_base}/ocr_results_full.json", "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)

#     print(f"\n[INFO] OCR results saved to {drive_base}/ocr_results_full.json")

"""
Spec-Lens — main entry point
=============================
Run this file to execute the full pipeline:
    Stage 1 — OCR:       video → frames → OCR text (EasyOCR or DeepSeek)
    Stage 2 — Reasoning: frames + OCR text → User Stories, Gherkin, Gap Analysis (LLaVA)

Colab setup (run ONCE in a separate cell before this script):
--------------------------------------------------------------
    # Only needed if using DeepSeek OCR engine:
    from src.deepseekocr_utils import engine as deepseek_engine
    deepseek_engine.download()
    # → Then restart the Colab runtime, and run this main.py cell.

VRAM note (T4 = 15 GB):
------------------------
    DeepSeek-OCR and LLaVA cannot both be loaded at the same time on a T4.
    This script frees OCR model memory before loading LLaVA automatically
    when ocr_engine = "deepseek".  EasyOCR is small enough that no manual
    unloading is needed.
"""

import gc
import json
import torch
from src.pipeline import process_video
from src.reasoning_pipeline import run_reasoning
from src.io_utils import save_json


# ── Configuration ──────────────────────────────────────────────────────────────

VIDEO_PATH = "data/raw/Simple_Bank_Transaction.mp4"

# Google Drive base folder — all outputs are persisted here across Colab sessions
DRIVE_BASE = "/content/drive/MyDrive/Spec-Lens-data"

# OCR engine: "easyocr" (default, lighter) or "deepseek" (higher accuracy)
OCR_ENGINE = "easyocr"

# Extract 1 frame every N frames from the video
FRAME_INTERVAL = 30

# Set to True to delete old extracted frames and start fresh
CLEAR_FRAMES = False

# Set to True to skip frames that already have a saved checkpoint
RESUME = True

# Limit frames processed — useful for quick testing. Set to None for full run.
MAX_FRAMES = None

# Set to True to run the LLaVA reasoning layer after OCR
RUN_REASONING = True


# ── Stage 1: OCR ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STAGE 1 — OCR")
print("=" * 60)

ocr_results = process_video(
    video_path=VIDEO_PATH,
    output_folder=f"{DRIVE_BASE}/frames",
    checkpoint_base_folder=f"{DRIVE_BASE}/ocr_results",
    interval=FRAME_INTERVAL,
    clear_frames=CLEAR_FRAMES,
    ocr_engine=OCR_ENGINE,
    resume=RESUME,
    max_frames=MAX_FRAMES,
)

# Print OCR results summary
for frame_result in ocr_results:
    frame = frame_result["frame"]
    print(f"\n── Frame {frame['frame_index']} (t={frame['timestamp_s']}s) ──")

    if frame_result["error"]:
        print(f"  [ERROR] {frame_result['error']}")
        continue

    print("  RAW OCR:")
    for item in frame_result["ocr_results"]:
        flag = "LOW" if item["is_low_confidence"] else "OK "
        print(f"    [{flag}] {item['text']} ({item['confidence']:.2f})")

    print("  CLEANED:")
    for text in frame_result["cleaned_text"]:
        print(f"    {text}")

# Save full OCR results to Drive
ocr_output_path = f"{DRIVE_BASE}/ocr_results_full.json"
save_json(ocr_results, ocr_output_path)
print(f"\n[INFO] OCR results saved → {ocr_output_path}")


# ── Free OCR model VRAM before loading LLaVA (required for DeepSeek on T4) ───

if RUN_REASONING and OCR_ENGINE == "deepseek":
    print("\n[INFO] Freeing DeepSeek VRAM before loading LLaVA...")
    from src.deepseekocr_utils import engine as deepseek_engine
    deepseek_engine.unload() if hasattr(deepseek_engine, "unload") else None
    gc.collect()
    torch.cuda.empty_cache()
    print("[INFO] VRAM cleared.")


# ── Stage 2: Reasoning ─────────────────────────────────────────────────────────

if RUN_REASONING:
    print("\n" + "=" * 60)
    print("STAGE 2 — REASONING (LLaVA-1.6)")
    print("=" * 60)

    reasoning_results = run_reasoning(
        ocr_results=ocr_results,
        checkpoint_folder=f"{DRIVE_BASE}/reasoning_results",
        resume=RESUME,
    )

    # Print reasoning results summary
    for result in reasoning_results:
        frame = result["frame"]
        print(f"\n── Frame {frame['frame_index']} (t={frame['timestamp_s']}s) ──")

        if result.get("error"):
            print(f"  [ERROR] {result['error']}")
            continue

        print(f"  USER STORY:\n    {result['user_story']}")
        print(f"  ACCEPTANCE CRITERIA:\n    {result['acceptance_criteria']}")
        print(f"  GAP ANALYSIS:\n    {result['gap_analysis']}")

    # Save full reasoning results to Drive
    reasoning_output_path = f"{DRIVE_BASE}/reasoning_results_full.json"
    save_json(reasoning_results, reasoning_output_path)
    print(f"\n[INFO] Reasoning results saved → {reasoning_output_path}")