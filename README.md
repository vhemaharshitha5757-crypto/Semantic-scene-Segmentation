# Offroad Semantic Segmentation using DINOv2

## Overview

This project performs semantic segmentation on off-road driving scenes using a frozen DINOv2 Vision Transformer backbone and a custom ConvNeXt-style segmentation head.

The model predicts pixel-level semantic classes for terrain understanding in autonomous navigation and robotics applications.

The segmentation head is trained on top of DINOv2 patch embeddings and achieves:

* **Validation Mean IoU:** 0.4586
* **Validation Dice Score:** 0.6253
* **Validation Pixel Accuracy:** 0.8218

---

# Project Structure

```text
mrdu_hackathon/
│
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   │
│   └── val/
│       ├── Color_Images/
│       └── Segmentation/
│
├── Offroad_Segmentation_testImages/
│   └── Color_Images/
│
└── Offroad_Segmentation_Scripts/
    └── Offroad_Segmentation_Scripts/
        ├── train_segmentation.py
        ├── test_segmentation.py
        ├── segmentation_head.pth
        ├── train_stats/
        └── predictions/
```

---

# Model Architecture

## Backbone

The feature extractor uses:

* DINOv2 Small (ViT-S/14)
* Embedding dimension: 384
* Frozen during training

Patch tokens are extracted using:

```python
backbone_model.forward_features(imgs)["x_norm_patchtokens"]
```

---

## Segmentation Head

Custom ConvNeXt-inspired decoder:

```text
Patch Tokens
      │
      ▼
1×1 Conv Projection
      │
      ▼
Residual Block 1
      │
      ▼
Residual Block 2
      │
      ▼
Residual Block 3
      │
      ▼
2× Bilinear Upsampling
      │
      ▼
1×1 Classification Layer
      │
      ▼
Segmentation Logits
```

Features:

* 1×1 Projection Layer
* Group Normalization
* GELU Activation
* Three Depthwise-Separable Residual Blocks
* Bilinear Upsampling
* Final Pixel Classification Layer

---

# Classes

The dataset contains 10 semantic classes.

Raw dataset mask values are converted into class IDs using:

```python
value_map
```

which maps original dataset labels into:

```text
0 - 9
```

class IDs used during training and evaluation.

---

# Training Configuration

| Parameter        | Value                    |
| ---------------- | ------------------------ |
| Backbone         | DINOv2 Small             |
| Batch Size       | 2                        |
| Learning Rate    | 1e-3                     |
| Optimizer        | AdamW                    |
| Weight Decay     | 1e-4                     |
| Scheduler        | CosineAnnealingLR        |
| Epochs           | 2                        |
| Loss Function    | CrossEntropy + Dice Loss |
| Input Resolution | 266 × 476                |

---

# Loss Function

Combined loss:

```text
Loss =
0.5 × CrossEntropyLoss
+
0.5 × DiceLoss
```

Benefits:

* CrossEntropy improves pixel classification.
* Dice Loss improves overlap quality.
* Produces better IoU performance than CrossEntropy alone.

---

# Training

Run training:

```bash
cd Offroad_Segmentation_Scripts/Offroad_Segmentation_Scripts

python3 train_segmentation.py
```

Outputs:

```text
segmentation_head.pth
train_stats/
```

Generated statistics:

* Loss Curves
* IoU Curves
* Dice Curves
* Accuracy Curves
* Evaluation Report

---

# Evaluation

Evaluate on validation dataset:

```bash
python3 test_segmentation.py \
    --data_dir ~/projects/mrdu_hackathon/Offroad_Segmentation_Training_Dataset/val \
    --model_path segmentation_head.pth
```

Outputs:

```text
predictions/
├── masks/
├── masks_color/
├── comparisons/
├── evaluation_metrics.txt
└── per_class_metrics.png
```

---

# Results

Final Validation Results

```text
Validation Loss      : 0.5027
Validation IoU       : 0.4586
Validation Dice      : 0.6253
Validation Accuracy  : 0.8218
```

Per-Epoch Results

```text
Epoch 1
Val IoU   : 0.4395

Epoch 2
Val IoU   : 0.4586
```

---

# Important Evaluation Fix

During development an evaluation bug was discovered.

## Problem

The evaluation script originally transformed segmentation masks using:

```python
transforms.ToTensor()
```

which is intended for images rather than semantic labels.

This corrupted mask values and caused validation IoU to be reported incorrectly.

Observed behavior:

```text
Training Validation IoU : 0.4586
Evaluation IoU          : 0.3280
```

---

## Fix

Evaluation masks were changed to use the same preprocessing pipeline as training:

```python
mask = TF.resize(
    mask,
    [266, 476],
    interpolation=TF.InterpolationMode.NEAREST
)

mask = torch.from_numpy(np.array(mask)).long()
```

After applying the fix:

```text
Training Validation IoU : 0.4586
Evaluation IoU          : 0.4586
```

This confirms consistency between training and evaluation.

---

# Dependencies

Install required packages:

```bash
pip install torch torchvision
pip install matplotlib
pip install numpy
pip install pillow
pip install opencv-python
pip install tqdm
```

---

# Reproducibility

Model checkpoint included:

```text
segmentation_head.pth
```

Performance reproduced:

```text
Mean IoU : 0.4586
Dice     : 0.6253
Accuracy : 0.8218
```

on the validation dataset.

---

# Author

VEERAMANENI HEMA HARSHITA

Semantic Segmentation of Off-Road Environments using DINOv2 and ConvNeXt-style Decoding.
