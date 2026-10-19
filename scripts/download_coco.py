import json
import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

TRAIN_JSON = "dataset/json/train.json"
VAL_JSON = "dataset/json/val.json"

OUTPUT_DIR = "dataset"

# if your internet connection is slow, try reducing the number like 16 or less
NUM_WORKERS = 32

CLASS_MAPPING = {
    0: 0,    # person
    25: 1,   # umbrella
    39: 2,   # bottle
    41: 3,   # cup
    42: 4,   # fork
    44: 5,   # spoon
    56: 6,   # chair
    67: 7,   # cell phone
    73: 8,   # book
}


def create_directories():

    for split in ["train", "val"]:

        os.makedirs(
            f"{OUTPUT_DIR}/images/{split}",
            exist_ok=True
        )

        os.makedirs(
            f"{OUTPUT_DIR}/labels/{split}",
            exist_ok=True
        )


def download_one(data, split):

    filename = data["file"]
    url = data["url"]

    image_path = os.path.join(
        OUTPUT_DIR,
        "images",
        split,
        filename
    )

    label_path = os.path.join(
        OUTPUT_DIR,
        "labels",
        split,
        os.path.splitext(filename)[0] + ".txt"
    )

    # if already downloaded
    if os.path.exists(image_path):
        return True

    try:

        response = requests.get(
            url,
            timeout=30
        )

        response.raise_for_status()

        with open(image_path, "wb") as f:
            f.write(response.content)

        # create YOLO label
        lines = []

        boxes = data.get(
            "annotations",
            {}
        ).get(
            "boxes",
            []
        )

        for box in boxes:

            coco_class = int(box[0])

            if coco_class not in CLASS_MAPPING:
                continue

            yolo_class = CLASS_MAPPING[coco_class]

            x_center = box[1]
            y_center = box[2]
            width = box[3]
            height = box[4]

            lines.append(
                f"{yolo_class} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        return True

    except Exception as e:

        return False


def process_split(json_file, split):

    with open(
        json_file,
        "r",
        encoding="utf-8"
    ) as f:

        images = json.load(f)

    print(
        f"\nDownloading {len(images):,} {split} images..."
    )

    success = 0
    failed = 0

    with ThreadPoolExecutor(
        max_workers=NUM_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                download_one,
                data,
                split
            )
            for data in images
        ]

        with tqdm(
            total=len(futures),
            unit="img"
        ) as pbar:

            for future in as_completed(futures):

                if future.result():
                    success += 1
                else:
                    failed += 1

                pbar.update(1)

    print(f"\nDownloaded: {success:,}")
    print(f"Failed:     {failed:,}")


create_directories()

process_split(
    TRAIN_JSON,
    "train"
)

process_split(
    VAL_JSON,
    "val"
)

print("\nDataset download complete!")