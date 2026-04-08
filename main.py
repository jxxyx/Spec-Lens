from src.pipeline import process_video
import json

if __name__ == "__main__":
    video_path = "data/raw/video1.mp4"

    drive_base = "/content/drive/MyDrive/Spec-Lens-data"

    results = process_video(
        video_path=video_path,
        output_folder=f"{drive_base}/frames",
        checkpoint_base_folder=f"{drive_base}/ocr_results",
        clear_frames=False,
        resume=True,
        max_frames=None
    )

    with open(f"{drive_base}/ocr_results_full.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n[INFO] OCR results saved to {drive_base}/ocr_results_full.json")