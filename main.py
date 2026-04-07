#To run the code to test if the skeleton works

#Import function from src
# To run the code to test if the skeleton works

from src.pipeline import process_video
import json

if __name__ == "__main__":
    video_path = "data/raw/video1.mp4"

    results = process_video(video_path)

    for frame_result in results:
        print(f"\nFrame: {frame_result['frame']}")

        print("\nRAW OCR:")
        for item in frame_result["ocr_results"]:
            flag = "LOW CONFIDENCE" if item["is_low_confidence"] else "OK"
            print(f"[{flag}] {item['text']} ({item['confidence']:.2f})")

        print("\nCLEANED:")
        for text in frame_result["cleaned_text"]:
            print(text)

    with open("data/ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n[INFO] OCR results saved to data/ocr_results.json")