import cv2 as cv
import time

from scripts.detector import ObjectDetector


def main():

    detector = ObjectDetector(
        model_path="models/yolo11n_coco9_best.pt",
        confidence=0.5
    )

    # Open webcam
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    # Camera resolution
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    previous_time = time.perf_counter()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read camera frame")
            break

        # Object detection
        results = detector.detect(frame)

        # Draw bounding boxes
        frame = detector.draw_boxes(frame, results)

        # Calculate FPS
        current_time = time.perf_counter()

        fps = 1 / max(current_time - previous_time, 1e-6)

        previous_time = current_time

        # Display FPS
        cv.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # Display camera feed
        cv.imshow("YOLO Object Detection", frame)

        # Press Q to quit
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()