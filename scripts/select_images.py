import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NDJSON_FILE = ROOT / "coco2017.ndjson"
OUTPUT_DIR = ROOT / "dataset" / "json"

TRAIN_COUNT = 40000
VAL_COUNT = 3000

random.seed(42)

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

train_images = []
val_images = []

with open(NDJSON_FILE, "r", encoding="utf-8") as f:

    for line in f:

        data = json.loads(line)

        if data.get("type") != "image":
            continue

        boxes = data.get("annotations", {}).get("boxes", [])

        # Keep only images containing at least one target class
        has_target = any(
            box[0] in target_classes
            for box in boxes
        )

        if not has_target:
            continue

        if data["split"] == "train":
            train_images.append(data)

        elif data["split"] == "val":
            val_images.append(data)


print(f"Available train images: {len(train_images):,}")
print(f"Available val images:   {len(val_images):,}")


# shuffle
random.shuffle(train_images)
random.shuffle(val_images)


# select subset
train_selected = train_images[:TRAIN_COUNT]
val_selected = val_images[:VAL_COUNT]


# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# save selected metadata
train_output = OUTPUT_DIR / "train.json"
val_output = OUTPUT_DIR / "val.json"

with train_output.open("w", encoding="utf-8") as f:
    json.dump(train_selected, f)

with val_output.open("w", encoding="utf-8") as f:
    json.dump(val_selected, f)


print("\nSelection complete!")

print(f"Selected train images: {len(train_selected):,}")
print(f"Selected val images:   {len(val_selected):,}")

print("\nSaved:")
print(train_output)
print(val_output)