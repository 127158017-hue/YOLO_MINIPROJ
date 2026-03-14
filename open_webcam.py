import sys
import os

# We are now inside the ultralytics_HTH directory.
sys.path.insert(0, './')

from ultralytics import YOLO

def main():
    print("Loading Custom YOLOv5s Model (GhostNet + CA + BiFPN)...")
    
    # The user provided a custom-trained model file 'best1.pt'
    # located at 'runs/detect/train/weights/best1.pt'
    weights_path = 'runs/detect/train/weights/best1.pt'
    
    if not os.path.exists(weights_path):
        print(f"Error: Could not find custom weights at {weights_path}.")
        return
        
    print(f"Loading weights from: {weights_path}")
    
    try:
        model = YOLO(weights_path)
        print("\nModel loaded successfully! Opening WebCam...")
        
        # Run inference on the primary webcam (source=0)
        # show=True will display the live window
        results = model.predict(source=0, show=True)
        
    except Exception as e:
        import traceback
        print(f"\nError during webcam inference: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
