import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# config

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"

SAMPLES_PER_SPLIT = 20

CLASS_NAMES = {
    0: "person",
    1: "umbrella",
    2: "bottle",
    3: "cup",
    4: "fork",
    5: "spoon",
    6: "chair",
    7: "cell phone",
    8: "book",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# check image - label pairs

def check_split(split):

    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    output_dir = DATASET_DIR / "dataset_check" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"\n========== {split.upper()} ==========")
    print(f"Images found: {len(images):,}")

    missing_labels = []
    invalid_labels = []
    empty_labels = []

    valid_images = []

    for image_path in images:

        label_path = label_dir / f"{image_path.stem}.txt"

        # check if labels exist
        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue

        # reading the labels
        try:

            with open(label_path, "r", encoding="utf-8") as f:
                lines = [
                    line.strip()
                    for line in f
                    if line.strip()
                ]

            if not lines:
                empty_labels.append(image_path.name)
                continue

            image_valid = True

            for line_number, line in enumerate(lines, start=1):

                parts = line.split()

                # YOLO format must contain 5 values
                if len(parts) != 5:
                    invalid_labels.append(
                        f"{label_path.name}: line {line_number}"
                    )
                    image_valid = False
                    continue

                try:
                    class_id = int(parts[0])

                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                except ValueError:

                    invalid_labels.append(
                        f"{label_path.name}: line {line_number}"
                    )
                    image_valid = False
                    continue

                # check class ID
                if class_id not in CLASS_NAMES:

                    invalid_labels.append(
                        f"{label_path.name}: invalid class {class_id}"
                    )

                    image_valid = False

                # check coordinates
                values = [x, y, w, h]

                if not all(0 <= value <= 1 for value in values):

                    invalid_labels.append(
                        f"{label_path.name}: invalid coordinates"
                    )

                    image_valid = False

                # Width and height must be > 0
                if w <= 0 or h <= 0:

                    invalid_labels.append(
                        f"{label_path.name}: invalid width/height"
                    )

                    image_valid = False

            if image_valid:
                valid_images.append(image_path)

        except Exception as e:

            invalid_labels.append(
                f"{label_path.name}: {e}"
            )

    # RESULTS

    print(f"Valid image/label pairs: {len(valid_images):,}")
    print(f"Missing labels:           {len(missing_labels):,}")
    print(f"Empty labels:             {len(empty_labels):,}")
    print(f"Invalid labels:           {len(invalid_labels):,}")

    if missing_labels:

        print("\nExamples of missing labels:")

        for item in missing_labels[:10]:
            print("  ", item)

    if invalid_labels:

        print("\nExamples of invalid labels:")

        for item in invalid_labels[:10]:
            print("  ", item)

    # VISUAL CHECK
    if not valid_images:

        print("\nNo valid images available for visualization.")
        return

    sample_count = min(
        SAMPLES_PER_SPLIT,
        len(valid_images)
    )

    samples = random.sample(
        valid_images,
        sample_count
    )

    print(
        f"\nCreating {sample_count} visual samples..."
    )

    for image_path in samples:

        label_path = label_dir / f"{image_path.stem}.txt"

        draw_image(
            image_path,
            label_path,
            output_dir
        )

    print(
        f"Saved visual samples to: {output_dir}"
    )


# DRAW BOUNDING BOXES
def draw_image(image_path, label_path, output_dir):

    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            16
        )
    except:
        font = ImageFont.load_default()

    with open(label_path, "r", encoding="utf-8") as f:

        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    for line in lines:

        parts = line.split()

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # Convert normalized YOLO coordinates to pixel coordinates

        x_center *= width
        y_center *= height

        box_width *= width
        box_height *= height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)

        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        class_name = CLASS_NAMES[class_id]


        # draw bounding box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        # draw label
        text = class_name

        bbox = draw.textbbox(
            (x1, y1),
            text,
            font=font
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle(
            [
                x1,
                y1 - text_height - 4,
                x1 + text_width + 6,
                y1
            ],
            fill="red"
        )

        draw.text(
            (x1 + 3, y1 - text_height - 2),
            text,
            fill="white",
            font=font
        )

    output_path = output_dir / image_path.name

    image.save(output_path)



if __name__ == "__main__":

    random.seed(42)

    check_split("train")
    check_split("val")

    print("\n====================================")
    print("Dataset check complete!")
    print("====================================")