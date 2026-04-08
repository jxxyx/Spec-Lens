from src.pipeline import process_video
import json

if __name__ == "__main__":
    video_path = "data/raw/video1.mp4"

    # Google Drive base folder for persistent outputs in Colab
    drive_base = "/content/drive/MyDrive/Spec-Lens-data"

    # ------------------------------------------
    # RESUME TEST SETTINGS
    # ------------------------------------------
    # FIRST TEST RUN:
    # clear_frames = True
    # resume = True
    # max_frames = 3
    #
    # SECOND TEST RUN:
    # clear_frames = False
    # resume = True
    # max_frames = None
    # ------------------------------------------

    clear_frames = False
    resume = True
    max_frames = None

    results = process_video(
        video_path=video_path,
        output_folder=f"{drive_base}/frames",
        checkpoint_base_folder=f"{drive_base}/ocr_results",
        interval=30,
        clear_frames=clear_frames,
        ocr_engine="easyocr",
        resume=resume,
        max_frames=max_frames
    )

    for frame_result in results:
        print(f"\nFrame: {frame_result['frame']}")

        print("\nRAW OCR:")
        for item in frame_result["ocr_results"]:
            flag = "LOW CONFIDENCE" if item["is_low_confidence"] else "OK"
            print(f"[{flag}] {item['text']} ({item['confidence']:.2f})")

        print("\nCLEANED:")
        for text in frame_result["cleaned_text"]:
            print(text)

    with open(f"{drive_base}/ocr_results_full.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n[INFO] OCR results saved to {drive_base}/ocr_results_full.json")