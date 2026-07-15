# Tutorial: Training Your Custom YOLOv5 Architecture on Kaggle

Now that we have successfully modified the YOLOv5 architecture to include a **GhostNet Backbone**, **Coordinate Attention (CA)**, and a **BiFPN Neck**, you have a state-of-the-art, ultra-lightweight model.

Since training a model on a large dataset requires a GPU to be reasonably fast, this tutorial explains how to take this local project, upload it to Kaggle (which provides free advanced GPUs), and train your model on your real dataset.

---

## Step 1: Prepare Your Local Environment

Before leaving your computer, you need to package the `ultralytics_HTH` folder.
Because we made structural modifications to the source code (in `block.py`, `conv.py`, `__init__.py`, and `tasks.py`), you cannot just `pip install ultralytics` on Kaggle. You must use _this exact folder_.

1. On your Windows machine, navigate to `D:\mini-project\trial_2\`.
2. Right-click the `ultralytics_HTH` folder.
3. Select **Compress to ZIP file** (or use 7-Zip/WinRAR).
4. Save the archive as `ultralytics_HTH.zip`.

## Step 2: Upload Files to Kaggle

1. Go to [kaggle.com](https://www.kaggle.com/) and sign in.
2. In the left-hand menu, click **Datasets** -> **New Dataset**.
3. Name it something like `Custom-YOLO-Source-Code`.
4. Upload your `ultralytics_HTH.zip` file.
5. (If your image dataset is not already on Kaggle, repeat this process to upload a `.zip` of all your training images and labels).

## Step 3: Create a Kaggle Notebook

1. Go to the Kaggle **Code** tab and click **New Notebook**.
2. On the right-hand panel, click **Add Data**. Search for and add the two datasets you just uploaded:
   - Your `Custom-YOLO-Source-Code` dataset.
   - Your image training dataset.
3. Open the **Settings** panel on the right side. Change the **Accelerator** from `None` to `GPU T4 x2` or `GPU P100`. (This is crucial for fast training).

## Step 4: Install Your Custom Ultralytics Package

In the very first cell of your Kaggle notebook, run the following commands to unzip your codebase and install it in "editable" mode so Python uses your modified files.

```bash
# Cell 1
!cp -r /kaggle/input/custom-yolo-source-code/ultralytics_HTH /kaggle/working/
%cd /kaggle/working/ultralytics_HTH
!pip install -e .
```

## Step 5: Start Training!

Create a new cell in your notebook. You can now instantiate the model dynamically using the YAML config we built, point it to your dataset, and train it on the Kaggle GPUs!

```python
# Cell 2
from ultralytics import YOLO

# Load our custom architecture setup
model = YOLO("ultralytics/cfg/models/v5/yolov5-ghost-ca-bifpn.yaml")

# Start training. Make sure 'data' points to YOUR yaml file describing your dataset
results = model.train(
    data="/kaggle/input/your-image-dataset-name/data.yaml",
    epochs=100,
    imgsz=640,
    device=0,  # Tells PyTorch to use the Kaggle GPU
)
```

## Step 6: Download Your Results

When the training is complete:

1. Navigate to your Kaggle Notebook's output directory.
2. Go to `/kaggle/working/ultralytics_HTH/runs/detect/train/weights/`.
3. Download the `best.pt` file. This is the holy grail containing all the intelligence your model just learned.

## Step 7: Run Live Inference Locally

Now that you have your powerful `best.pt` weights, go back to your local Windows computer.

1. Move the `best.pt` file into `D:\mini-project\trial_2\ultralytics_HTH\`.
2. Open powershell and activate your environment `.\venv\Scripts\activate`.
3. You can either use the `open_webcam.py` we created earlier, or run standard predictions targeting the new weights:

```python
from ultralytics import YOLO

# Load your newly trained weights
model = YOLO("best.pt")

# Run the webcam live feed
model.predict(source=0, show=True)
```

You have now successfully trained and deployed a custom, research-grade YOLOv5 model!
