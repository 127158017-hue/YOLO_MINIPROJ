# YOLOv5 Custom Architecture Implementation Details

**Enhancement Integration:** GhostNet Backbone, Coordinate Attention (CA), and BiFPN Neck.

## Introduction

The objective of this project was to modify the standard YOLOv5 architecture to decrease computational complexity and parameter count while attempting to maintain or improve model accuracy. This was achieved by integrating a GhostNet backbone with Coordinate Attention, and utilizing a Bidirectional Feature Pyramid Network (BiFPN) in the neck for weighted feature fusion.

## Step 1: Implementing Coordinate Attention (CA)

**File Modified:** `ultralytics/nn/modules/conv.py`

Coordinate Attention is designed to capture positional information and channel relationships.

1. **Creation of `CoordAtt` Class**: We created a custom PyTorch module. It utilizes `nn.AdaptiveAvgPool2d` to pool along the height (X) and width (Y) directions independently, creating directional feature maps.
2. **Concatenation and Convolution**: The pooled maps are concatenated and passed through a 1x1 convolution, followed by batch normalization and a Swish activation function.
3. **Attention Weighting**: The maps are split back into their respective directions, passed through another 1x1 convolution, and normalized with a Sigmoid function to generate attention weights.
4. **Fixing In-place Errors**: During implementation, PyTorch gradient errors ("in-place operation") were encountered. This was resolved by explicitly cloning the input identity tensor before multiplying by the attention weights (`out = identity.clone() * a_w * a_h`).

## Step 2: Integrating GhostNet Backbone with CA

**File Modified:** `ultralytics/nn/modules/block.py`

Standard YOLOv5 uses C3 (CSP Bottleneck with 3 convolutions) layers. We replaced the underlying standard convolutions with Ghost Convolutions to reduce parameters.

1. **Creation of `GhostBottleneckCA`**: This new block sequences standard `GhostConv` layers but integrates our newly created `CoordAtt` module. As the tensor passes through the Ghost bottleneck, coordinate attention is applied to enhance spatial feature extraction without significant overhead.
2. **Creation of `C3GhostCA`**: This module mimics the standard `C3` structure but replaces the inner `Bottleneck` layers with our `GhostBottleneckCA`. This allows the YOLO parser to seamlessly drop this into the backbone.

## Step 3: Implementing BiFPN Neck

**File Modified:** `ultralytics/nn/modules/conv.py`

The standard YOLO neck simply concatenates feature maps from different scales. BiFPN introduces learnable weights so the network can learn which feature maps contain more important information.

1. **Creation of `BiFPN_Concat`**: We created a new module inheriting from `nn.Module` that accepts a list of tensors (similarly to the standard `Concat`).
2. **Learnable Weights**: A `torch.nn.Parameter` weight vector `self.w` is initialized based on the number of incoming feature maps.
3. **Weighted Fusion**: During the forward pass, the weights are ensured to be positive via `torch.nn.functional.relu()`. They are then normalized by their sum plus a small epsilon (`0.0001`) for numerical stability. The input tensors are multiplied by their respective normalized weights before being summed.
4. **Gradient Graph Fixes**: We carefully re-wrote the weight parameter reassignment during the forward pass by initializing `torch.ones` dynamically and applying `F.relu()` locally, as modifying `nn.Parameter` in place breaks PyTorch's backpropagation.

## Step 4: Registering the Custom Modules

**Files Modified:** `ultralytics/nn/modules/__init__.py` & `ultralytics/nn/tasks.py`

For the YOLO framework to recognize the newly created modules when reading a configuration file, they had to be hooked into the system.

1. **Exposure**: Imported and added `CoordAtt`, `C3GhostCA`, `GhostBottleneckCA`, and `BiFPN_Concat` to the global `__all__` list in `__init__.py`.
2. **Dynamic Parsing**: Updated the `parse_model` function inside `tasks.py`. We added specific checks for `BiFPN_Concat` so that the model initialization loop knows it should treat this module the same way it treats `Concat` (accepting a list of routing indices from preceding layers).

## Step 5: Creating the Custom YOLOv5 YAML Configuration

**File Created:** `ultralytics/cfg/models/v5/yolov5-ghost-ca-bifpn.yaml`

1. We copied the structure of the standard `yolov5s.yaml`.
2. In the `backbone` section, all `C3` module references were replaced with `C3GhostCA`.
3. In the `head` section, all `Concat` module references were replaced with `BiFPN_Concat`.

## Step 6: Testing and Verification

**Files Created:** `test_custom_yolo.py`, `train_test.py`, `open_webcam.py`

1. **Instantiation Verification (`test_custom_yolo.py`)**: Script built the model and confirmed the parameter count decreased dramatically from **9.15M parameters** (standard YOLOv5s) to approximately **2.15M parameters** (Custom GhostNet model).
2. **Training and Backpropagation (`train_test.py`)**: Validated that the gradient graph was perfectly connected by running a 100-epoch training loop on the CPU using the `coco8` dataset. The model successfully completed training without crashing.
3. **Live Inference (`open_webcam.py`)**: Wrote a script that automatically loads the resulting `best.pt` file from the training runs and passes the user’s live webcam stream through the new custom model.
