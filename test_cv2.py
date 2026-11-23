import os
import time

import cv2

# Prevent GUI hangs
os.environ["DISPLAY"] = ""


def test_camera():
    print("Testing OpenCV VideoCapture across possible RealSense nodes...")

    # Try the video nodes listed for RealSense in v4l2-ctl
    for node in [0, 4, 2, 1]:
        print(f"Trying /dev/video{node}...")

        # Open with explicit V4L2 API
        cap = cv2.VideoCapture(node, cv2.CAP_V4L2)

        # Force format and size before reading
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print(f"Node {node} failed to open.")
            continue

        print(f"Node {node} opened successfully. Attempting to read frame...")

        # Read a few frames to let camera warm up
        for _ in range(5):
            ret, frame = cap.read()
            if ret:
                break
            time.sleep(0.1)

        if ret:
            print(f"SUCCESS: Read frame of shape {frame.shape} from /dev/video{node}")
            cap.release()
            return True
        else:
            print(f"Node {node} opened, but could not read frame.")
            cap.release()

    print("All RealSense nodes failed to read frames.")
    return False


if __name__ == "__main__":
    test_camera()
