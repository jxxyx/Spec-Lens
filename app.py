import gc
import json
import torch
from src.pipeline import process_video
from src.reasoning_pipeline import run_reasoning
from src.io_utils import save_json
import gradio as gr

def run_pipeline(video_path: str) -> list[dict]:
    VIDEO_PATH = video_path

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
    reasoning_results = []

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

            if result.get("truncated"):
                print("  [WARNING] Response was truncated — token budget exceeded.")

            print(f"\n  SCREEN SUMMARY:\n    {result['screen_summary']}")

            for us in result.get("user_stories", []):
                print(f"\n  ── {us['title']} ──")

                print("\n  ROCSTAR DETAILS:")
                print(f"    {us['story']}")

                if us.get("assembled_story"):
                    print("\n  USER STORY:")
                    print(f"    {us['assembled_story']}")

                if us.get("scenario"):
                    print(f"\n  SCENARIO:")
                    print(f"    {us['scenario']}")

            print(f"\n  ACCEPTANCE CRITERIA:\n    {result['acceptance_criteria']}")
            print(f"\n  GAP ANALYSIS:\n    {result['gap_analysis']}")

        # Save full reasoning results to Drive
        reasoning_output_path = f"{DRIVE_BASE}/reasoning_results_full.json"
        save_json(reasoning_results, reasoning_output_path)
        print(f"\n[INFO] Reasoning results saved → {reasoning_output_path}")

    return reasoning_results

def format_results_for_display(reasoning_results: list[dict]) -> str:
    """
    Convert machine-readable reasoning results into a human-readable report
    for display in Gradio.
    """

    if not reasoning_results:
        return "No reasoning results generated."

    report_sections = []

    for idx, result in enumerate(reasoning_results, start=1):
        frame = result.get("frame", {})
        frame_index = frame.get("frame_index", "Unknown")
        timestamp = frame.get("timestamp_s", "Unknown")

        section = []

        section.append(
            f"=== Frame {idx} | Original Frame: {frame_index} | Timestamp: {timestamp}s ==="
        )

        if result.get("error"):
            section.append(f"\nERROR:\n{result['error']}")
            report_sections.append("\n".join(section))
            continue

        section.append("\nSCREEN SUMMARY:")
        section.append(result.get("screen_summary", "N/A"))

        section.append("\nUSER STORIES:")

        for us_idx, us in enumerate(result.get("user_stories", []), start=1):

            section.append(f"\nUS-{us_idx}: {us.get('title', 'Untitled')}")

            if us.get("assembled_story"):
                section.append("\nUser Story:")
                section.append(us["assembled_story"])

            if us.get("story"):
                section.append("\nROCSTAR Details:")
                section.append(us["story"])

            if us.get("scenario"):
                section.append("\nScenario:")
                section.append(us["scenario"])

        section.append("\nACCEPTANCE CRITERIA:")
        section.append(result.get("acceptance_criteria", "N/A"))

        section.append("\nGAP ANALYSIS:")
        section.append(result.get("gap_analysis", "N/A"))

        if result.get("validation"):
            validation = result["validation"]

            section.append("\nVALIDATION:")
            section.append(f"Valid: {validation.get('valid', False)}")

            errors = validation.get("errors", [])

            if errors:
                section.append("Errors:")

                for err in errors:
                    section.append(f"- {err}")

        report_sections.append("\n".join(section))

    # Separator between every frame
    return ("\n" + "=" * 100 + "\n").join(report_sections)


def run_app(video_path):
    """
    Wrapper function for Gradio.

    Takes the uploaded video path, runs the full Spec-Lens pipeline,
    and returns a formatted text report for display.
    """
    if video_path is None:
        return "Please upload a video first."

    reasoning_results = run_pipeline(video_path)
    return format_results_for_display(reasoning_results)


with gr.Blocks() as demo:
    gr.Markdown("# Spec-Lens: Multimodal BA Requirements Generator")

    video_input = gr.Video(label="Upload UI Workflow Video")

    run_button = gr.Button("Run Spec-Lens")

    output_box = gr.Textbox(
        label="Generated BA Artefacts",
        lines=30,
        max_lines=50,
    )

    run_button.click(
        fn=run_app,
        inputs=video_input,
        outputs=output_box,
    )


if __name__ == "__main__":
    demo.launch(share=True)