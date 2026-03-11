import sys
import os

# We are now inside the ultralytics_HTH directory.
sys.path.insert(0, './')

from ultralytics import YOLO

def main():
    print("Loading Custom YOLOv5s Model (GhostNet + CA + BiFPN)...")
    
    # Path to the best weights from the recent 100-epoch training run.
    # Ultralytics saves the latest run in 'runs/detect/trainX/weights/best.pt'.
    # We will search for the latest 'train' directory.
    
    runs_dir = '../runs/detect'
    if not os.path.exists(runs_dir):
        print(f"Error: Could not find training runs at {runs_dir}.")
        print("Please train the model first.")
        return
        
    train_dirs = [d for d in os.listdir(runs_dir) if d.startswith('train')]
    if not train_dirs:
        print("Error: No training directories found.")
        return
        
    # Sort to get the latest train directory
    latest_train_dir = sorted(train_dirs, key=lambda x: int(x.replace('train', '0')) if x != 'train' else 0)[-1]
    weights_path = os.path.join(runs_dir, latest_train_dir, 'weights', 'best.pt')
    
    if not os.path.exists(weights_path):
        print(f"Error: Could not find weights at {weights_path}.")
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
