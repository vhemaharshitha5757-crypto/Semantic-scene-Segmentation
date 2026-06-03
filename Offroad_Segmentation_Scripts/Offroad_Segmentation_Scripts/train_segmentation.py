"""
Segmentation Training Script
Converted from train_mask.ipynb
Trains a segmentation head on top of DINOv2 backbone
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os
import random
import torchvision
from tqdm import tqdm

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = len(value_map)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Joint Augmentation (applied identically to image + mask)
# ============================================================================

class JointTransform:
    """Applies the same spatial transform to both image and mask."""
    def __init__(self, image_size):
        self.image_size = image_size  # (H, W)

    def __call__(self, image, mask):
        h, w = self.image_size

        # Resize
        image = TF.resize(image, [h, w], interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  [h, w], interpolation=TF.InterpolationMode.NEAREST)

        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # Random rotation (small angle)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.rotate(mask,  angle, interpolation=TF.InterpolationMode.NEAREST)

        # Color jitter on image only
        if random.random() > 0.5:
            image = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            )(image)

        # To tensor + normalize image
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Mask to tensor (keep as integer class ids)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask


class ValTransform:
    """Deterministic transform for validation (no augmentation)."""
    def __init__(self, image_size):
        self.image_size = image_size  # (H, W)

    def __call__(self, image, mask):
        h, w = self.image_size

        image = TF.resize(image, [h, w], interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  [h, w], interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, joint_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.joint_transform = joint_transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path  = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path)
        mask  = convert_mask(mask)

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        return image, mask


# ============================================================================
# Loss: Combined Cross-Entropy + Dice (greatly improves IoU)
# ============================================================================

class DiceLoss(nn.Module):
    """Soft Dice loss over all classes."""
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W)   targets: (B, H, W) long
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes)          # (B,H,W,C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()   # (B,C,H,W)

        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        cardinality  = (probs + targets_one_hot).sum(dim=(0, 2, 3))

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """0.5 * CrossEntropy + 0.5 * Dice — balances per-pixel and per-class accuracy."""
    def __init__(self, num_classes, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.ce_w   = ce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        return self.ce_w * self.ce(logits, targets) + self.dice_w * self.dice(logits, targets)


# ============================================================================
# Model: Improved Segmentation Head
# ============================================================================

class ResidualBlock(nn.Module):
    """Depthwise-separable residual block (ConvNeXt-style)."""
    def __init__(self, channels):
        super().__init__()
        self.dw   = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.pw1  = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.pw2  = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.act  = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x + residual


class SegmentationHeadConvNeXt(nn.Module):
    """
    Improved head:
      1. Project patch tokens → 256-ch feature map at patch resolution
      2. Three residual blocks for richer features
      3. 2× bilinear upsample → classify at 2× patch resolution
         (keeps spatial detail before final F.interpolate to full res)
    """
    def __init__(self, in_channels, out_channels, tokenW, tokenH, hidden=256):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        # Project from embedding dim to hidden channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GroupNorm(num_groups=min(32, hidden), num_channels=hidden),
            nn.GELU(),
        )

        # Three residual blocks
        self.res1 = ResidualBlock(hidden)
        self.res2 = ResidualBlock(hidden)
        self.res3 = ResidualBlock(hidden)

        # 2× upsample then classify
        self.upsample   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.classifier = nn.Conv2d(hidden, out_channels, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)   # (B,C,H,W)
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.upsample(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_inds   = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union        = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class)


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Dice coefficient (F1 Score) per class and return mean Dice Score."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds   = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score   = (2. * intersection + smooth) / (
            pred_inds.sum().float() + target_inds.sum().float() + smooth
        )
        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10, show_progress=True):
    """Evaluate all metrics on a dataset."""
    iou_scores       = []
    dice_scores      = []
    pixel_accuracies = []

    model.eval()
    loader = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") if show_progress else data_loader
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            output  = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits  = model(output.to(device))
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            # labels already long integers (no squeeze needed since JointTransform returns 2-D mask)
            if labels.dim() == 3:
                pass  # (B, H, W) — correct
            else:
                labels = labels.squeeze(dim=1)

            iou       = compute_iou(outputs, labels, num_classes=num_classes)
            dice      = compute_dice(outputs, labels, num_classes=num_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)

    model.train()
    return np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies)


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    print(f"Saved training curves to '{output_dir}/training_curves.png'")

    # Plot 2: IoU curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.title('Train IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('Validation IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'))
    plt.close()
    print(f"Saved IoU curves to '{output_dir}/iou_curves.png'")

    # Plot 3: Dice curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.title('Train Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Validation Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'))
    plt.close()
    print(f"Saved Dice curves to '{output_dir}/dice_curves.png'")

    # Plot 4: Combined metrics plot
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['train_iou'], label='train')
    plt.plot(history['val_iou'], label='val')
    plt.title('IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['train_dice'], label='train')
    plt.plot(history['val_dice'], label='val')
    plt.title('Dice Score vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()
    print(f"Saved combined metrics curves to '{output_dir}/all_metrics_curves.png'")


def save_history_to_file(history, output_dir):
    """Save training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")

        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_iou'][i],
                history['val_iou'][i],
                history['train_dice'][i],
                history['val_dice'][i],
                history['train_pixel_acc'][i],
                history['val_pixel_acc'][i]
            ))

    print(f"Saved evaluation metrics to {filepath}")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 2
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)
    lr       = 1e-3          # AdamW works well at 1e-3 with cosine decay
    n_epochs = 2

    # Output directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # Dataset paths (relative to script location)
    data_dir = os.path.join(script_dir, '..', '..', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir  = os.path.join(script_dir, '..', '..', 'Offroad_Segmentation_Training_Dataset', 'val')

    # Joint transforms (spatial ops applied identically to image + mask)
    train_transform = JointTransform(image_size=(h, w))
    val_transform   = ValTransform(image_size=(h, w))

    # Create datasets
    trainset    = MaskDataset(data_dir=data_dir, joint_transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)

    valset    = MaskDataset(data_dir=val_dir, joint_transform=val_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Training samples:   {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    # Load DINOv2 backbone
    print("Loading DINOv2 backbone...")
    BACKBONE_SIZE = "small"
    backbone_archs = {
        "small": "vits14",
        "base":  "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    backbone_arch  = backbone_archs[BACKBONE_SIZE]
    backbone_name  = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)
    print("Backbone loaded successfully!")

    # Get embedding dimension from backbone
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")
    print(f"Patch tokens shape:  {output.shape}")

    # Create segmentation head (improved: 3 residual blocks + 2× upsample)
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14,
        hidden=256,
    )
    classifier = classifier.to(device)

    # ── Loss: Combined CE + Dice ──────────────────────────────────────────────
    loss_fct = CombinedLoss(num_classes=n_classes, ce_weight=0.5, dice_weight=0.5)

    # ── Optimizer: AdamW (better weight decay handling than SGD) ─────────────
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)

    # ── Scheduler: CosineAnnealingLR (smooth LR decay over all epochs) ───────
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * len(train_loader), eta_min=1e-6
    )

    # Training history
    history = {
        'train_loss':      [],
        'val_loss':        [],
        'train_iou':       [],
        'val_iou':         [],
        'train_dice':      [],
        'val_dice':        [],
        'train_pixel_acc': [],
        'val_pixel_acc':   []
    }

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        # ── Training phase ────────────────────────────────────────────────────
        classifier.train()
        train_losses = []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]",
                          leave=False, unit="batch")
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

            logits  = classifier(output.to(device))
            # Upsample to full image resolution using bilinear (labels stay integer)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            # labels: (B, H, W) long — already correct from JointTransform
            loss = loss_fct(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()   # step per batch for cosine annealing

            train_losses.append(loss.item())
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # ── Validation phase ──────────────────────────────────────────────────
        classifier.eval()
        val_losses = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]",
                        leave=False, unit="batch")
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs, labels = imgs.to(device), labels.to(device)

                output  = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                logits  = classifier(output.to(device))
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                loss = loss_fct(outputs, labels)
                val_losses.append(loss.item())
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Compute epoch metrics ─────────────────────────────────────────────
        train_iou, train_dice, train_pixel_acc = evaluate_metrics(
            classifier, backbone_model, train_loader, device, num_classes=n_classes
        )
        val_iou, val_dice, val_pixel_acc = evaluate_metrics(
            classifier, backbone_model, val_loader, device, num_classes=n_classes
        )

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss   = np.mean(val_losses)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_pixel_acc)
        history['val_pixel_acc'].append(val_pixel_acc)

        epoch_pbar.set_postfix(
            train_loss=f"{epoch_train_loss:.3f}",
            val_loss=f"{epoch_val_loss:.3f}",
            val_iou=f"{val_iou:.3f}",
            val_acc=f"{val_pixel_acc:.3f}"
        )

    # Save plots
    print("\nSaving training curves...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)

    # Save model (in scripts directory)
    model_path = os.path.join(script_dir, "segmentation_head.pth")
    torch.save(classifier.state_dict(), model_path)
    print(f"Saved model to '{model_path}'")

    # Final evaluation
    print("\nFinal evaluation results:")
    print(f"  Final Val Loss:     {history['val_loss'][-1]:.4f}")
    print(f"  Final Val IoU:      {history['val_iou'][-1]:.4f}")
    print(f"  Final Val Dice:     {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Accuracy: {history['val_pixel_acc'][-1]:.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
