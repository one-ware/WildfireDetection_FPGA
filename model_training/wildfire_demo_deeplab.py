import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


# =========================
# Dataset
# =========================

class SegmentationDataset(Dataset):
    def __init__(self, base_dir, split="Train", img_wh=(640, 360)):
        self.img_dir = os.path.join(base_dir, split)
        self.img_wh = img_wh

        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(".png") and not f.endswith("_seg.png")
        ])

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        label_name = img_name.replace(".png", "_seg.png")

        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        label = Image.open(os.path.join(self.img_dir, label_name)).convert("RGBA")

        img = img.resize(self.img_wh, Image.BILINEAR)
        label = label.resize(self.img_wh, Image.NEAREST)

        # Alpha -> binary mask
        alpha = np.asarray(label)[:, :, 3]
        mask = (alpha > 127).astype(np.float32)

        img = np.asarray(img, dtype=np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))

        mask = np.expand_dims(mask, 0)

        return torch.tensor(img), torch.tensor(mask)


# =========================
# Model
# =========================

class DeepLabBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(weights="DEFAULT")

        in_ch = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x):
        return self.model(x)["out"]


# =========================
# Loss
# =========================

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = 1 - (2 * intersection + 1) / (probs.sum() + targets.sum() + 1)

        return bce + dice


# =========================
# Training
# =========================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total += loss.item() * x.size(0)

    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total += loss.item() * x.size(0)

    return total / len(loader.dataset)


# =========================
# ONNX Export
# =========================

def export_onnx(model, path, device, input_size=(1, 3, 360, 640)):
    model.eval()
    dummy = torch.randn(*input_size).to(device)

    torch.onnx.export(
        model,
        dummy,
        path,
        opset_version=18,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
    )

    print("Saved ONNX:", path)


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.join(BASE_DIR, "../dataset")
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--export_onnx", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SegmentationDataset(args.dataset_dir, "Train")
    val_ds = SegmentationDataset(args.dataset_dir, "Validation")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True   # Important for stable batch shapes during training
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        drop_last=False
    )

    model = DeepLabBinary().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = DiceBCELoss()

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    if args.export_onnx:
        export_onnx(model, "deeplab_binary.onnx", device)


if __name__ == "__main__":
    main()