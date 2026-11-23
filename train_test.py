import os
import sys

# We are now inside the directory.
sys.path.insert(0, "./")

from ultralytics import YOLO


def main():
    print("Initializing Custom YOLOv5s (GhostNet + CA + BiFPN)...")

    # Instantiate the model with our custom configuration
    config_path = "ultralytics/cfg/models/v5/yolov5-ghost-ca-bifpn.yaml"

    if not os.path.exists(config_path):
        print(f"Error: Could not find config at {config_path}")
        return

    try:
        model = YOLO(config_path)
        print("\nModel instantiated successfully. Starting test training with coco8 dataset...")

        # Train the model with a tiny dataset (coco8.yaml) for a few epochs
        # using a very small image size for maximum speed on a CPU.
        model.train(
            data="coco8.yaml",  # Built-in 8-image dataset
            epochs=100,  # User requested 100 epochs
            imgsz=320,  # Small image size
            device="cpu",  # Force CPU training explicitly
        )

        print("\nTraining completed successfully! The custom architecture is fully functional.")

    except Exception as e:
        import traceback

        print(f"\nError taking place during training initialization: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
