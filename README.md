# ObjectDetectionYolo

A computer-vision project for training a YOLO object detector on a curated
subset of COCO 2017 and running the resulting model on a live webcam feed.

## Project goal

This project explores the end-to-end process of building an object detector:
selecting classes from coco2017 dataset, preparing images and YOLO
labels, training and evaluating a model, and using it for real-time detection.
The included application displays the webcam image with detected object names,
confidence scores, bounding boxes, and an FPS estimate.

## Detectable classes

My model is trained to detect these nine COCO classes:

| Model class ID | Class |
| ---: | --- |
| 0 | Person |
| 1 | Umbrella |
| 2 | Bottle |
| 3 | Cup |
| 4 | Fork |
| 5 | Spoon |
| 6 | Chair |
| 7 | Cell phone |
| 8 | Book |

The model IDs are remapped from the original COCO category IDs when labels are
created. See `config/coco.yaml` for the names used during training.

## Repository contents

```text
config/
  coco.yaml                 YOLO dataset configuration
models/
  yolo11n_coco9_best.pt     Trained model used by the webcam application
notebooks/
  coco9_yolo.ipynb          Training, evaluation, and model export workflow
scripts/
  inspect_coco.py           Summarize target classes in the COCO metadata
  select_images.py          Select train and validation image metadata
  count_classes.py          Count target classes in the selected metadata
  download_coco.py          Download selected images and create YOLO labels
  check_dataset.py          Validate labels and create visual samples
  detector.py               YOLO tracking and bounding-box drawing helper
main.py                     Webcam object-detection application
```


## Requirements

- Python 3.10 or newer
- A webcam for live detection
- Internet access and sufficient disk space to download the dataset
- A CUDA-capable GPU is recommended for training; inference can also run on
  the device supported by your Ultralytics/PyTorch installation

## Installation

```
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python requests tqdm pillow pyyaml notebook
```


## Prepare the dataset

The scripts expect the COCO 2017 image metadata file named
`coco2017.ndjson` in the repository root. Obtain the COCO detection dataset
metadata from the
[Ultralytics COCO dataset documentation](https://docs.ultralytics.com/datasets/detect/coco/),
and place the NDJSON file at ROOT path.

## Run the python scripts
Before running, you can customize the classes, number of train/val images. \
And I also recommend adjusting the number of workers in download_coco.py relevant to your pc and internet stats. 

Run these commands from the repository root, in order:

```
python scripts/inspect_coco.py
python scripts/select_images.py
python scripts/count_classes.py
python scripts/download_coco.py
python scripts/check_dataset.py
```


The scripts create the following local dataset structure:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  json/
    train.json
    val.json
```

The label files use YOLO's normalized bounding-box format. The dataset folder
is excluded from Git; download and prepare it locally before training.

## Train and evaluate

Run the `notebooks/coco9_yolo.ipynb` file to train the model. My notebook uses
the pretrained `yolo11n.pt` model, with a 640-pixel image size, batch size 16,
and 50 training epochs by default. It first provides a short smoke test, then
trains and evaluates the model, and copies the best weights to
`models/yolo11n_coco9_best.pt`. \
(My pc has limited GPU and VRAM, and the best I can fine-tune is yolo11 Nano. You can expect better results on models with more parameters.) \
Training outputs are written under `runs/` and are excluded from Git.

## Run webcam detection

From the repository root, run:

```
python main.py
```

The application loads `models/yolo11n_coco9_best.pt`, opens the default
webcam, and displays detections and FPS in a window. Press **Q** to quit.
Connect a webcam before launching. If the default camera is not the one you
want, change the camera index in `cv.VideoCapture(0)` in `main.py`.
