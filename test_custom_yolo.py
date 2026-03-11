import sys
sys.path.insert(0, './')
from ultralytics import YOLO

def main():
    try:
        print("\n--- Standard YOLOv5s ---")
        model_standard = YOLO('yolov5su.yaml')
        model_standard.info()

        # Instantiate the model with our custom config
        print("\n--- Custom YOLOv5s (GhostNet + CA + BiFPN) ---")
        model_custom = YOLO('ultralytics/cfg/models/v5/yolov5-ghost-ca-bifpn.yaml')
        
        # Print model information (this will implicitly run Info() and show parameter count)
        print("\nModel instantiated successfully!")
        model_custom.info()
        
    except Exception as e:
        print(f"\nError instantiating model: {e}")

if __name__ == "__main__":
    main()
