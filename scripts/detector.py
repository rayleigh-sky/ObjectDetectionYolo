from ultralytics import YOLO
import cv2 as cv


class ObjectDetector:
    def __init__(self, model_path, confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.3,
            iou=0.7,
            max_det=50,
            imgsz=640,
            verbose=False
        )

        return results[0]

    def draw_boxes(self, frame, results):

        for box in results.boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Confidence score
            confidence = float(box.conf[0])

            # Class ID
            class_id = int(box.cls[0])

            # Class name
            class_name = results.names[class_id]

            # Label
            label = f"{class_name}: {confidence:.2f}"

            # Draw bounding box
            cv.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw label background
            (text_width, text_height), baseline = cv.getTextSize(
                label,
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            label_y = max(y1, text_height + baseline + 10)

            cv.rectangle(
                frame,
                (x1, label_y - text_height - baseline - 10),
                (x1 + text_width + 10, label_y),
                (0, 255, 0),
                -1
            )

            # Draw label
            cv.putText(
                frame,
                label,
                (x1 + 5, label_y - baseline - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame