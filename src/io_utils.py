import json
from pathlib import Path


def save_json(data, file_path: str) -> None:
    """Serialise data to a JSON file, creating parent directories as needed."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(file_path: str):
    """Load and return the contents of a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_exists(file_path: str) -> bool:
    """Return True if the given file path exists on disk."""
    return Path(file_path).exists()