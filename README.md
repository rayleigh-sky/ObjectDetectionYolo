# ObjectDetectionYolo

The goal of this project is to build a real-time object detection application 
using YOLO and a webcam. It captures live video, detects objects with a
custom-trained YOLO model, and displays the results with bounding boxes and
the current frames-per-second (FPS) rate. This project provides a practical
way to explore computer vision and apply object detection to live camera
footage.

## Dataset customization

The size of the coco2017 dataset is 27GB and downloading the whole dataset to fine-tune a yolo model is impractical. 
Instead I tried customizing the json file to download specific classes.
My classes are: 
  0: person
  1: umbrella
  2: bottle
  3: cup
  4: fork
  5: spoon
  6: chair
  7: cell phone
  8: book


## How to run this project?
