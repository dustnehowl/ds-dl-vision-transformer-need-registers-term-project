"""
=============================================================================
 Vision Transformers Need Registers — Table 2 ADE20k
 Phase 2 v2: Linear Semantic Segmentation (Official DINOv2 Protocol)

 Changes from v1 (phase2_ade20k.py):
   - 4-layer feature extraction (architecture-specific block indices)
     instead of single last-layer patchtokens
   - BatchNorm2d before classification conv (following DINOv2 BNHead)
   - get_intermediate_layers(reshape=True) API for spatial features
   - Conv2d(4*D → 150, kernel_size=1) head

 Reference: dinov2/eval/segmentation/models/decode_heads/linear_head.py
            dinov2/notebooks/semantic_segmentation.ipynb

 Protocol:
   - Frozen backbone (DINOv2-L)
   - 4 intermediate layers concatenated → BN → Conv2d(4D → 150)
   - Bilinear upsample to image resolution
   - Train 20k iter on ADE20k training set
   - Eval mIoU on validation set (single-scale)
=============================================================================
"""

import os
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm


# ADE20K Challenge: 150 valid classes (1..150), class 0 = void/ignore
NUM_CLASSES = 150
IGNORE_INDEX = 255

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_layer_indices(model_name):
    """DINOv2 paper-specified intermediate layer indices (0-indexed).
    Same indices used for both segmentation and depth probes.
    Reference: dinov2/hub/depthers.py out_index tables."""
    name = model_name.lower()
    if "vitl" in name:
        return [4, 11, 17, 23]
    if "vitg" in name:
        return [9, 19, 29, 39]
    if "vits" in name or "vitb" in name:
        return [2, 5, 8, 11]
    return None  # fallback handled in model init


# ============================================================================
# Dataset
# ============================================================================

class ADE20K(Dataset):
    """ADEChallengeData2016 dataset.

    Dir structure:
        root/images/{training,validation}/<id>.jpg
        root/annotations/{training,validation}/<id>.png
    Annotations: 0 = void/ignore, 1..150 = classes
    """
    def __init__(self, root, split, image_size, augment=False):
        self.image_dir = Path(root) / "images" / split
        self.ann_dir = Path(root) / "annotations" / split
        self.image_files = sorted(self.image_dir.glob("*.jpg"))
        self.image_size = image_size
        self.augment = augment
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No images in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def _transform(self, img, ann):
        S = self.image_size
        if self.augment:
            # Random scale in [0.5, 2.0]
            scale = float(np.random.uniform(0.5, 2.0))
            w, h = img.size
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            img = img.resize((nw, nh), Image.BICUBIC)
            ann = ann.resize((nw, nh), Image.NEAREST)

            # Pad if smaller than crop size
            pad_w = max(0, S - nw)
            pad_h = max(0, S - nh)
            if pad_w or pad_h:
                img = TF.pad(img, [0, 0, pad_w, pad_h], fill=0)
                ann = TF.pad(ann, [0, 0, pad_w, pad_h], fill=0)
                nw += pad_w
                nh += pad_h

            # Random crop S x S
            x0 = int(np.random.randint(0, nw - S + 1))
            y0 = int(np.random.randint(0, nh - S + 1))
            img = img.crop((x0, y0, x0 + S, y0 + S))
            ann = ann.crop((x0, y0, x0 + S, y0 + S))

            # Random horizontal flip
            if np.random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                ann = ann.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            # Validation: simple resize to S x S
            img = img.resize((S, S), Image.BICUBIC)
            ann = ann.resize((S, S), Image.NEAREST)

        # To tensors
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1)
        for c in range(3):
            img_t[c] = (img_t[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        ann_arr = np.asarray(ann, dtype=np.int64)
        ann_t = torch.from_numpy(ann_arr.copy())
        # Remap: 0 -> IGNORE, 1..150 -> 0..149
        ann_t = torch.where(ann_t == 0, torch.tensor(IGNORE_INDEX), ann_t - 1)
        return img_t, ann_t

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        ann_path = self.ann_dir / (img_path.stem + ".png")
        img = Image.open(img_path).convert("RGB")
        ann = Image.open(ann_path)
        return self._transform(img, ann)


# ============================================================================
# Model: frozen backbone + 4-layer BNHead segmentation
#
# Official DINOv2 segmentation head (linear_head.py BNHead):
#   1. get_intermediate_layers(n=layer_indices, reshape=True)
#      → list of [B, D, H_p, W_p], one per layer
#   2. Resize all layers to same spatial size → concat → [B, 4D, H_p, W_p]
#   3. SyncBatchNorm(4D)
#   4. Conv2d(4D → num_classes, kernel_size=1)
# ============================================================================

class LinearSegmenter4Layer(nn.Module):
    def __init__(self, backbone, embed_dim, layer_indices,
                 num_classes=NUM_CLASSES, patch_size=14):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.layer_indices = list(layer_indices)
        self.n_layers = len(self.layer_indices)
        self.patch_size = patch_size

        in_channels = self.n_layers * embed_dim
        # Official BNHead: SyncBatchNorm → Conv2d(1x1)
        # (use BatchNorm2d for single-GPU, functionally equivalent)
        self.bn = nn.BatchNorm2d(in_channels)
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        with torch.no_grad():
            # Official API: reshape=True returns [B, D, H_p, W_p] per layer
            # norm defaults to True (applies final LayerNorm)
            # return_class_token=False (default for segmentation)
            feats = self.backbone.get_intermediate_layers(
                x, n=self.layer_indices,
                reshape=True,
                return_class_token=False,
            )
        # feats: list of [B, D, H_p, W_p], all same spatial size
        feat = torch.cat(feats, dim=1)    # [B, n_layers*D, H_p, W_p]

        # BN + Conv
        feat = self.bn(feat)
        logits = self.head(feat)           # [B, K, H_p, W_p]

        # Upsample to input resolution
        logits = F.interpolate(logits, size=(H, W),
                               mode="bilinear", align_corners=False)
        return logits


# ============================================================================
# Evaluation: mIoU via confusion matrix
# ============================================================================

def compute_miou(confusion):
    tp = confusion.diag().float()
    fp = confusion.sum(0).float() - tp
    fn = confusion.sum(1).float() - tp
    iou = tp / (tp + fp + fn).clamp(min=1)
    valid = (tp + fn) > 0
    return iou[valid].mean().item() * 100, iou


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device, dtype=torch.long)
    for img, ann in tqdm(val_loader, desc="eval", leave=False, dynamic_ncols=True):
        img = img.to(device, non_blocking=True)
        ann = ann.to(device, non_blocking=True)
        logits = model(img)
        pred = logits.argmax(dim=1)
        valid = ann != IGNORE_INDEX
        p = pred[valid]
        g = ann[valid]
        idx = g * NUM_CLASSES + p
        confusion += torch.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES) \
                          .reshape(NUM_CLASSES, NUM_CLASSES)
    miou, _ = compute_miou(confusion)
    return miou


# ============================================================================
# Training loop
# ============================================================================

def train_one(model_name, args, device):
    print(f"\n{'='*68}")
    print(f"Training segmentation head (4-layer BNHead): {model_name}")
    print('=' * 68)

    # Load backbone
    backbone = torch.hub.load("facebookresearch/dinov2", model_name)
    backbone = backbone.to(device).eval()
    embed_dim = backbone.embed_dim
    patch_size = backbone.patch_size

    # Determine layer indices
    if args.layer_indices is not None:
        layer_indices = args.layer_indices
        layer_source = "user-specified"
    else:
        layer_indices = get_layer_indices(model_name)
        if layer_indices is None:
            total = len(backbone.blocks)
            layer_indices = list(range(total - 4, total))
            layer_source = f"fallback last 4 (total={total})"
        else:
            layer_source = "DINOv2 paper"

    print(f"  embed_dim={embed_dim}, patch_size={patch_size}")
    print(f"  feature layers (0-idx): {layer_indices}  [{layer_source}]")
    print(f"  head input channels = {len(layer_indices) * embed_dim} "
          f"({len(layer_indices)} layers x {embed_dim})")

    model = LinearSegmenter4Layer(
        backbone, embed_dim, layer_indices, NUM_CLASSES, patch_size
    ).to(device)

    # Data
    train_ds = ADE20K(args.ade20k_root, "training", args.image_size, augment=True)
    val_ds   = ADE20K(args.ade20k_root, "validation", args.image_size, augment=False)
    print(f"  train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Optimizer: BN params + head conv params
    trainable_params = list(model.bn.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(trainable_params,
                                   lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_iter)

    # Training
    model.bn.train()
    model.head.train()
    iter_loader = iter(train_loader)
    t0 = time.time()
    pbar = tqdm(range(args.n_iter), desc="train", dynamic_ncols=True)
    running_loss = None

    for step in pbar:
        try:
            img, ann = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            img, ann = next(iter_loader)

        img = img.to(device, non_blocking=True)
        ann = ann.to(device, non_blocking=True)

        logits = model(img)
        loss = F.cross_entropy(logits, ann, ignore_index=IGNORE_INDEX)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        l = loss.item()
        running_loss = l if running_loss is None else 0.99 * running_loss + 0.01 * l
        if step % 50 == 0:
            pbar.set_postfix({
                "loss": f"{running_loss:.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        if step > 0 and step % args.eval_every == 0 and step < args.n_iter - 1:
            miou = evaluate(model, val_loader, device)
            tqdm.write(f"  [step {step}/{args.n_iter}] mIoU = {miou:.2f}%")
            model.bn.train()
            model.head.train()

    # Final eval
    final_miou = evaluate(model, val_loader, device)
    elapsed = (time.time() - t0) / 60
    print(f"\n  -> {model_name} final mIoU = {final_miou:.2f}%   ({elapsed:.1f} min)")

    del backbone, model
    torch.cuda.empty_cache()
    return final_miou


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ade20k_root", default="/nas/datahub/ADEChallengeData2016")
    p.add_argument("--models", nargs="+",
                   default=["dinov2_vitl14", "dinov2_vitl14_reg"])
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--n_iter", type=int, default=20000)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--layer_indices", type=int, nargs="+", default=None,
                   help="Override 0-indexed block indices to concat. "
                        "Default: auto from DINOv2 paper "
                        "(ViT-L -> [4,11,17,23], ViT-S/B -> [2,5,8,11])")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", default="./results_phase2_ade20k_v2.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"Device: {device}")
    print(f"ADE20k root: {args.ade20k_root}")
    print(f"Image size: {args.image_size}, batch_size: {args.batch_size}, "
          f"iter: {args.n_iter}")
    print(f"Layer indices: {args.layer_indices or 'auto (paper default)'}")

    results = {}
    for m in args.models:
        try:
            results[m] = round(train_one(m, args, device), 2)
        except Exception as e:
            print(f"  ERROR for {m}: {e}")
            import traceback; traceback.print_exc()
            results[m] = None
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*64}")
    print("FINAL — ADE20k Semantic Segmentation (mIoU, %)")
    print('=' * 64)
    for m, v in results.items():
        display = f"{v:.2f}" if v is not None else "FAILED"
        print(f"  {m:32}  {display:>8}")

    valid = [v for v in results.values() if v is not None]
    if len(valid) == 2:
        diff = valid[1] - valid[0]
        print(f"\n  Delta (reg - no_reg) = {diff:+.2f}")
        print(f"  Paper Table 2 reports: 46.6 -> 47.9  (Delta = +1.3)")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
