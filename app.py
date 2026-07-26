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

def format_results_as_chat_bubbles(reasoning_results: list[dict]) -> list[list[str]]:
    """Convert reasoning results into a chat-style list-of-lists for Gradio Chatbot.

    Each item is a two-element list: [speaker, message].
    """

    bubbles: list[list[str]] = []

    if not reasoning_results:
        return [["Spec-Lens", "No reasoning results generated."]]

    for idx, result in enumerate(reasoning_results, start=1):
        speaker = f"Frame {idx}"

        frame = result.get("frame", {})
        frame_index = frame.get("frame_index", "Unknown")
        timestamp = frame.get("timestamp_s", "Unknown")
        header = f"Original Frame: {frame_index} | Timestamp: {timestamp}s"

        if result.get("error"):
            bubbles.append([speaker, f"{header}\nERROR: {result['error']}"])
            continue

        parts: list[str] = [header, "SCREEN SUMMARY:", result.get("screen_summary", "N/A")]

        user_stories = result.get("user_stories", [])
        if user_stories:
            parts.append("USER STORIES:")
            for us_idx, us in enumerate(user_stories, start=1):
                parts.append(f"{us_idx}. {us.get('title', 'Untitled')}")
                if us.get("assembled_story"):
                    parts.append(f"   • User Story: {us['assembled_story']}")
                if us.get("story"):
                    parts.append(f"   • ROCSTAR Details: {us['story']}")
                if us.get("scenario"):
                    parts.append(f"   • Scenario: {us['scenario']}")

        parts.append("ACCEPTANCE CRITERIA:")
        parts.append(result.get("acceptance_criteria", "N/A"))
        parts.append("GAP ANALYSIS:")
        parts.append(result.get("gap_analysis", "N/A"))

        if result.get("validation"):
            validation = result["validation"]
            parts.append("VALIDATION:")
            parts.append(f"   • Valid: {validation.get('valid', False)}")
            errors = validation.get("errors", [])
            if errors:
                parts.append("   • Errors (full):")
                for err in errors:
                    # Support structured error items (dict/list) and plain strings
                    if isinstance(err, (dict, list)):
                        err_str = json.dumps(err, ensure_ascii=False)
                    else:
                        err_str = str(err)
                    parts.append(f"     - {err_str}")

        bubbles.append([speaker, "\n".join(parts)])

    return bubbles


def run_app(video_path):
    if video_path is None:
        return "❌ Please upload a video first.", []

    if isinstance(video_path, dict):
        video_path = video_path.get("path") or video_path.get("video")

    reasoning_results = run_pipeline(video_path)

    # Collect validation failures (frame index, timestamp, brief errors)
    invalid_count = 0
    failed_frames = []
    for r in reasoning_results:
        v = r.get("validation")
        if v is not None and not v.get("valid", False):
            invalid_count += 1
            frame = r.get("frame", {})
            frame_index = frame.get("frame_index", "Unknown")
            timestamp = frame.get("timestamp_s")
            ts_str = f"t={timestamp}s" if timestamp is not None else "t=?s"

            errors = v.get("errors", []) if isinstance(v, dict) else []
            if errors:
                # Join up to first 3 error messages to keep status concise
                err_summary = "; ".join(errors[:3])
                if len(errors) > 3:
                    err_summary += " (and more)"
            else:
                err_summary = "validation failed"

            failed_frames.append(f"Frame {frame_index} ({ts_str}): {err_summary}")

    if invalid_count > 0:
        failure_details = "; ".join(failed_frames)
        status_message = (
            f"❌ Spec-Lens completed with validation failures: {invalid_count} of {len(reasoning_results)} screens failed validation. "
            f"Failed frames: {failure_details}"
        )
    else:
        status_message = (
            f"✅ Spec-Lens completed successfully. "
            f"Generated BA artefacts for {len(reasoning_results)} unique screens."
        )

    chat_bubbles = format_results_as_chat_bubbles(reasoning_results)

    return status_message, chat_bubbles


with gr.Blocks() as demo:
    gr.Markdown("# Spec-Lens: Multimodal BA Requirements Generator")
    gr.Markdown("Upload a workflow video and click **Send** to process it. Each frame will appear as its own assistant bubble.")

    video_input = gr.Video(label="Upload UI Workflow Video")

    run_button = gr.Button("Send")

    status_box = gr.Textbox(
        label="Status",
        interactive=False,
    )

    chatbot = gr.Chatbot(elem_id="spec_lens_chat", label="Spec-Lens Chat Output")

    # Inject JS to auto-scroll the chatbot container to the bottom when new messages arrive
    gr.HTML(
        """
        <script>
        (function(){
          const waitFor = (fn, timeout=5000) => {
            const start = Date.now();
            const check = () => {
              const el = document.getElementById('spec_lens_chat');
              if (el) { fn(el); return; }
              if (Date.now() - start < timeout) requestAnimationFrame(check);
            };
            check();
          };

          waitFor(function(chat){
            // chat may contain an inner scrollable div; fallback to chat element
            const container = chat.querySelector('.scrollable') || chat;
            const observer = new MutationObserver(()=>{ container.scrollTop = container.scrollHeight; });
            observer.observe(container, {childList:true, subtree:true});
            // initial scroll
            container.scrollTop = container.scrollHeight;
          });
        })();
        </script>
        """
    )

    run_button.click(
        fn=run_app,
        inputs=video_input,
        outputs=[status_box, chatbot],
    )


demo.launch(share=True)