import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ndjson_file_path = ROOT / "coco2017.ndjson"

if not ndjson_file_path.exists():
    raise FileNotFoundError(f"COCO dataset not found at: {ndjson_file_path}")

# COCO class IDs we want
target_classes = {
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

counts = Counter()

train_images = []
val_images = []

if ndjson_file_path.exists():
    with ndjson_file_path.open("r", encoding="utf-8") as f:

        for line in f:
            data = json.loads(line)

            if data.get("type") != "image":
                continue

            boxes = data.get("annotations", {}).get("boxes", [])

            # find which target classes occur in this image
            found_classes = set()

            for box in boxes:
                class_id = box[0]

                if class_id in target_classes:
                    found_classes.add(class_id)

            # image contains at least one of our classes
            if found_classes:

                for class_id in found_classes:
                    counts[class_id] += 1

                if data["split"] == "train":
                    train_images.append(data)

                elif data["split"] == "val":
                    val_images.append(data)


print("\n========== RESULTS ==========\n")

print(f"Train images containing target classes: {len(train_images):,}")
print(f"Validation images containing target classes: {len(val_images):,}")

print("\nImages per class:")

for class_id, class_name in target_classes.items():
    print(f"{class_id:2}  {class_name:12} : {counts[class_id]:,}")