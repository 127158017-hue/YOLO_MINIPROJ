# `open_webcam.py` Documentation

This document describes the functionality and usage of the `open_webcam.py` script.

## Overview

The `open_webcam.py` script is designed to perform real-time object detection and distance measurement using an Intel RealSense camera and a YOLO object detection model. It captures both RGB (color) and depth streams from the RealSense camera, aligns them, runs YOLO inference to detect objects, and then calculates the real-world distance to the highest-confidence detected object.

## Dependencies

Ensure the following Python libraries are installed before running the script:

- `pyrealsense2`: For interacting with Intel RealSense cameras.
- `numpy`: For array manipulations and mathematical operations.
- `opencv-python` (`cv2`): For image processing and displaying the output window.
- `ultralytics`: For running the YOLO object detection model.

## Configuration Parameters

At the top of the script, several configuration variables are defined:

- `MODEL_PATH`: The absolute path to the trained YOLO model file (e.g., `best1.pt`). You must update this path to point to your specific model.
- `CONF_THRESHOLD`: The minimum confidence score (default is `0.3`) required to consider a detection valid.
- `WIDTH` / `HEIGHT`: The resolution of the camera streams (default `424x240`). A smaller resolution is used to ensure high-speed processing, particularly useful for constrained edge devices like the Raspberry Pi 5.

## Script Workflow

1. **Initialize YOLO**:
   The script loads the specified YOLO model and retrieves the dictionary of class names the model was trained on.

2. **Configure RealSense Pipeline**:
   It initializes the RealSense pipeline and configures both the color (`rs.stream.color`) and depth (`rs.stream.depth`) streams to the designated resolution and framerate (30 FPS). The streams are started, and an alignment object is created to map depth data to the color frames.

3. **Frame Processing Loop**:
   Inside an infinite `while` loop, the script performs the following operations:
   - Waits for the next set of frames from the camera.
   - Aligns the depth frame to the color frame to ensure depth data corresponds precisely to the visual pixels.
   - Converts the color frame into a NumPy array (`img`) compatible with OpenCV and YOLO.

4. **YOLO Inference**:
   The `img` array is passed to the YOLO model for inference, using an image size of `320` to optimize execution speed.

5. **Filtering Detections**:
   The script iterates over all detected objects to find the single object with the highest confidence score (`max_conf`).

6. **Distance Calculation**:
   For the highest-confidence object:
   - It identifies the bounding box coordinates (`x1, y1, x2, y2`) and calculates the center point (`cx, cy`).
   - It samples the depth values in a 3x3 grid around the center point.
   - The median of these valid depth samples is computed to determine the final distance, filtering out potential noise.

7. **Display Output**:
   - A green bounding box is drawn around the detected object.
   - A label containing the object's class name, calculated distance in meters, and confidence score is overlaid on the image.
   - A red dot marks the center of the bounding box.
   - The processed frame is displayed in an OpenCV window named "RealSense Pi 5 Detection".

8. **Termination**:
   The script continuously runs until the user presses the `q` key. Upon termination, it safely stops the RealSense pipeline and closes all OpenCV windows to release resources.

## Usage

To use the script:

1. Connect your Intel RealSense camera.
2. Ensure the `MODEL_PATH` is appropriately set in the script.
3. Run the script using Python:
   ```bash
   python open_webcam.py
   ```
4. Press `q` while the output window is focused to exit the application.
