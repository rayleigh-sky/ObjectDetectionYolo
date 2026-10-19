import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "dataset" / "json"

CLASS_NAMES = {
    0: "person",
    25: "umbrella",
    39: "bottle",
    41: "cup",
    42: "fork",
    44: "spoon",
    56: "chair",
    67: "cell phone",
    73: "book",
}


def count_images_by_class(json_path: Path) -> Counter[int]:
    with json_path.open("r", encoding="utf-8") as file:
        images: list[dict[str, Any]] = json.load(file)

    counts: Counter[int] = Counter()
    for image in images:
        boxes = image.get("annotations", {}).get("boxes", [])
        image_class_ids = {int(box[0]) for box in boxes}
        counts.update(image_class_ids.intersection(CLASS_NAMES))

    return counts


def main() -> None:
    train_counts = count_images_by_class(JSON_DIR / "train.json")
    val_counts = count_images_by_class(JSON_DIR / "val.json")

    print(f"{'Class':<12} {'Train images':>12} {'Val images':>12}")
    print("-" * 38)
    for class_id, class_name in CLASS_NAMES.items():
        print(
            f"{class_name:<12} "
            f"{train_counts[class_id]:>12,} "
            f"{val_counts[class_id]:>12,}"
        )


if __name__ == "__main__":
    main()
