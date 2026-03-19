import os
import argparse
import numpy as np
from PIL import Image
from PIL import Image as PILImage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# Dataset
# =========================

class SegmentationDataset(Dataset):
    def __init__(self, base_dir, split='Train', img_size=(640, 360), invert_mask=False):
        self.img_dir = os.path.join(base_dir, split)
        self.img_size = img_size
        self.invert_mask = invert_mask

        self.img_files = [
            f for f in os.listdir(self.img_dir)
            if f.endswith('.png') and not f.endswith('_seg.png')
        ]
        self.img_files.sort()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        label_name = img_name.replace('.png', '_seg.png')

        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.img_dir, label_name)

        img = Image.open(img_path).convert('RGB').resize(self.img_size, Image.BILINEAR)
        label_img = Image.open(label_path).convert('RGBA').resize(self.img_size, Image.NEAREST)

        img_np = np.asarray(img, dtype=np.float32) / 255.0

        label_rgba = np.asarray(label_img, dtype=np.uint8)

        # Alpha channel is the mask
        alpha = label_rgba[:, :, 3]

        # Everything visible is foreground (= 1)
        label_np = (alpha > 0).astype(np.float32)

        if self.invert_mask:
            label_np = 1.0 - label_np

        img_np = np.transpose(img_np, (2, 0, 1))
        label_np = np.expand_dims(label_np, axis=0)

        return torch.from_numpy(img_np), torch.from_numpy(label_np)


# =========================
# UNet
# =========================

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for feature in features:
            self.downs.append(self.double_conv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.ups.append(self.double_conv(feature * 2, feature))

        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


# =========================
# Training
# =========================

def export_onnx(model, onnx_path, device, input_size=(1, 3, 360, 640), opset=17):
    model.eval()

    dummy_input = torch.randn(*input_size).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print(f"Saved ONNX model to: {onnx_path}")

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


# =========================
# Visualization
# =========================

def overlay_mask_on_image(image_chw, mask_hw):
    img = np.transpose(image_chw, (1, 2, 0))
    img_u8 = (img * 255).astype(np.uint8)

    mask_bin = (mask_hw >= 0.5)

    # Bright red
    img_u8[mask_bin] = [255, 0, 0]

    return img_u8


def save_visualization(img_t, label_t, pred_prob_hw, out_prefix, threshold):
    img_np = img_t.cpu().numpy()
    gt_hw = label_t.cpu().squeeze(0).numpy()

    pred_bin = (pred_prob_hw >= threshold).astype(np.uint8) * 255
    gt_bin = (gt_hw >= 0.5).astype(np.uint8) * 255

    overlay_gt = overlay_mask_on_image(img_np, gt_hw)
    overlay_pred = overlay_mask_on_image(img_np, pred_prob_hw)

    PILImage.fromarray((np.transpose(img_np, (1, 2, 0)) * 255).astype(np.uint8)).save(out_prefix + "_input.png")
    PILImage.fromarray(gt_bin.astype(np.uint8)).save(out_prefix + "_gt_mask.png")
    PILImage.fromarray(pred_bin.astype(np.uint8)).save(out_prefix + "_pred_mask.png")
    PILImage.fromarray(overlay_gt).save(out_prefix + "_overlay_gt.png")
    PILImage.fromarray(overlay_pred).save(out_prefix + "_overlay_pred.png")

    print("Saved visualization:", out_prefix)


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    parser.add_argument('--export_onnx', action='store_true')
    parser.add_argument('--onnx_path', type=str, default='unet_seg.onnx')
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), '../dataset')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = SegmentationDataset(base_dir, 'Train')
    val_ds = SegmentationDataset(base_dir, 'Validation')

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs} - Train: {train_loss:.4f} - Val: {val_loss:.4f}")

    # Automatically find a validation sample containing an object
    sample_index = None
    for i in range(len(val_ds)):
        _, label = val_ds[i]
        if label.sum() > 0:
            sample_index = i
            break

    if sample_index is None:
        print("WARNING: No object found in the validation split.")
        return

    img, label = val_ds[sample_index]

    model.eval()
    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits).cpu().squeeze().numpy()

    out_prefix = os.path.join(os.path.dirname(__file__), "val_sample")
    save_visualization(img, label, prob, out_prefix, args.conf_threshold)

    if args.export_onnx:
        export_onnx(
            model,
            args.onnx_path,
            device,
            opset=18,
            input_size=(1, 3, 360, 640)  # Adjust to your configured img_size.
        )


if __name__ == "__main__":
    main()