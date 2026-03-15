import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "/home/harin/miniproj/ultralytics_HTH/runs/detect/train/weights/best1.pt"  # <--- SET YOUR PATH HERE
   # Your custom model file
CONF_THRESHOLD = 0.3    # Minimum confidence to show a detection
WIDTH, HEIGHT = 424, 240 # Small, high-speed resolution for Pi 5

# 1. Initialize YOLO
print("Loading model and class names...")
model = YOLO(MODEL_PATH)
# This gets the dictionary of names: {0: 'person', 1: 'cup', ...}
class_names = model.names 

# 2. Configure RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

print("Starting Camera...")
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

try:
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert to numpy
        img = np.asanyarray(color_frame.get_data())

        # 3. YOLO Inference (imgsz=320 makes it even faster)
        results = model(img, conf=CONF_THRESHOLD, imgsz=320, verbose=False)

        # 4. Filter for the Highest Confidence detection
        best_box = None
        max_conf = 0
        best_class_id = 0

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    best_box = box
                    best_class_id = int(box.cls[0]) # Get the class index

        # 5. Get Name and Distance
        if best_box is not None:
            # Look up the actual name from the model's labels
            object_name = class_names[best_class_id]
            
            # Coordinates
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Calculate Distance (Median of 3x3 to filter noise)
            dist_samples = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    d = depth_frame.get_distance(cx + i, cy + j)
                    if d > 0: dist_samples.append(d)
            
            distance = np.median(dist_samples) if dist_samples else 0

            # 6. Display formatting
            color = (0, 255, 0) # Green box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Create label: "Name: Distance (Conf)"
            label = f"{object_name.capitalize()}: {distance:.2f}m ({max_conf:.2f})"
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            # Draw text
            cv2.putText(img, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw center point
            cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)

        # 7. Final Output
        cv2.imshow("RealSense Pi 5 Detection", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
