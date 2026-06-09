"""
=============================================================================
 Phase 3 v2: NYUd Depth — Official DINOv2 BNHead Protocol

 Changes from previous versions:
   - Consolidated from phase3_nyud_paper.py + phase3_nyud_hf.py
   - Exactly follows dinov2/hub/depthers.py + dinov2/hub/depth/decode_heads.py
   - Supports both .mat and HuggingFace H5 data formats (auto-detect)
   - Official BNHead architecture:
       * 4-layer extraction with return_class_token=True, norm=False
       * Per layer: concat patches [B,D,H,W] + broadcast CLS [B,D,H,W] → [B,2D,H,W]
       * Upsample each layer ×4, then concat → [B, 4×2D, H', W']
       * Conv2d(8D → 256 bins, kernel_size=1)
       * bins_strategy="UD" (linear spacing)
       * norm_strategy="linear" (ReLU + 0.1 eps + L1 normalize)
       * depth = einsum(probs, bin_centers)
   - Official SigLoss (AdaBins/Eigen 2014):
       L = sqrt(var(g) + 0.15 * mean(g)^2)  where g = log(pred) - log(gt)
       with loss_weight = 10.0
   - 30% linear warm-up → cosine annealing

 Reference:
   dinov2/hub/depthers.py            — model factory + layer indices
   dinov2/hub/depth/decode_heads.py  — BNHead, depth_pred, UD bins
   dinov2/eval/depth/models/losses/sigloss.py  — SigLoss

 Paper Target: RMSE 0.378 → 0.366 (Delta -0.012)
=============================================================================
"""

import os
import argparse
import json
import time
import re
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
from PIL import Image
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEPTH_MIN = 1e-3
DEPTH_MAX = 10.0
NUM_BINS = 256
HEAD_UPSAMPLE = 4          # spatial ×4 before head conv (official default)
NORM_EPS = 0.1             # ReLU + eps + L1-normalize (norm_strategy="linear")
SI_EPS = 0.001             # SigLoss numerical stability (official value)


def get_layer_indices(model_name):
    """DINOv2 paper-specified layer indices (0-indexed).
    Reference: dinov2/hub/depthers.py out_index tables."""
    name = model_name.lower()
    if "vitl" in name:
        return [4, 11, 17, 23]
    if "vitg" in name:
        return [9, 19, 29, 39]
    if "vits" in name or "vitb" in name:
        return [2, 5, 8, 11]
    return None


# ============================================================================
# Data loading: supports both .mat (labeled NYU) and HF H5 format
# ============================================================================

_NYU_CACHE = {}


def load_nyu_mat(mat_path):
    if mat_path in _NYU_CACHE:
        return _NYU_CACHE[mat_path]
    print(f"Loading NYU .mat: {mat_path}")
    with h5py.File(mat_path, "r") as f:
        images = f["images"][:]
        depths = f["depths"][:]
    print(f"  Raw shapes: images={images.shape}, depths={depths.shape}")

    if images.shape[0] != 1449:
        for ax in range(images.ndim):
            if images.shape[ax] == 1449:
                images = np.moveaxis(images, ax, 0)
                break
    if depths.shape[0] != 1449:
        for ax in range(depths.ndim):
            if depths.shape[ax] == 1449:
                depths = np.moveaxis(depths, ax, 0)
                break
    if images.shape[-1] != 3:
        for ax in range(1, images.ndim):
            if images.shape[ax] == 3:
                images = np.moveaxis(images, ax, -1)
                break
    if images.shape[1] == 640 and images.shape[2] == 480:
        images = np.transpose(images, (0, 2, 1, 3))
    if depths.ndim == 3 and depths.shape[1] == 640 and depths.shape[2] == 480:
        depths = np.transpose(depths, (0, 2, 1))

    print(f"  Final shapes: images={images.shape}, depths={depths.shape}")
    print(f"  Depth range: [{depths.min():.3f}, {depths.max():.3f}] m")
    _NYU_CACHE[mat_path] = (images.astype(np.uint8), depths.astype(np.float32))
    return _NYU_CACHE[mat_path]


def load_split_indices(split_path):
    with open(split_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    try:
        return [int(l) - 1 for l in lines]
    except ValueError:
        idxs = []
        for l in lines:
            m = re.search(r'(\d+)', l)
            if m:
                idxs.append(int(m.group(1)) - 1)
        return idxs


class NYUMatDataset(Dataset):
    """NYU depth from nyu_depth_v2_labeled.mat + train/test split files."""
    def __init__(self, mat_path, split_path, image_size=518, augment=False):
        self.images, self.depths = load_nyu_mat(mat_path)
        self.indices = load_split_indices(split_path)
        N = self.images.shape[0]
        self.indices = [i for i in self.indices if 0 <= i < N]
        self.image_size = image_size
        self.augment = augment
        print(f"  Split {Path(split_path).name}: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = self.images[idx]
        depth = self.depths[idx]
        return self._process(img, depth)

    def _process(self, img, depth):
        S = self.image_size
        H_orig, W_orig = img.shape[:2]

        if self.augment:
            scale = float(np.random.uniform(0.7, 1.3))
            nw, nh = max(S, int(W_orig * scale)), max(S, int(H_orig * scale))
            img_pil = Image.fromarray(img).resize((nw, nh), Image.BICUBIC)
            depth_t = torch.from_numpy(depth.copy()).unsqueeze(0).unsqueeze(0)
            depth_t = F.interpolate(depth_t, size=(nh, nw), mode="nearest").squeeze()

            x0 = np.random.randint(0, nw - S + 1)
            y0 = np.random.randint(0, nh - S + 1)
            img_pil = img_pil.crop((x0, y0, x0 + S, y0 + S))
            depth_t = depth_t[y0:y0 + S, x0:x0 + S]

            if np.random.random() < 0.5:
                img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                depth_t = torch.flip(depth_t, dims=[-1])

            img_arr = np.asarray(img_pil, dtype=np.float32) / 255.0
        else:
            img_pil = Image.fromarray(img).resize((S, S), Image.BICUBIC)
            img_arr = np.asarray(img_pil, dtype=np.float32) / 255.0
            depth_t = torch.from_numpy(depth.copy()).unsqueeze(0).unsqueeze(0)
            depth_t = F.interpolate(depth_t, size=(S, S), mode="nearest").squeeze()

        for c in range(3):
            img_arr[..., c] = (img_arr[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        img_t = torch.from_numpy(img_arr.transpose(2, 0, 1).copy())

        mask = (depth_t > DEPTH_MIN) & (depth_t < DEPTH_MAX)
        return img_t, depth_t, mask


class NYUH5Dataset(Dataset):
    """NYU depth from HuggingFace extracted H5 files.
    Layout:
        root/train/<scene_name>/<frame_id>.h5
        root/val/official/<frame_id>.h5
    Each .h5: rgb (3,480,640) uint8, depth (480,640) float32 meters.
    """
    def __init__(self, root, split="train", image_size=518, augment=False):
        self.image_size = image_size
        self.augment = augment

        if split == "train":
            patt = os.path.join(root, "train", "*", "*.h5")
        elif split == "val":
            patt = os.path.join(root, "val", "official", "*.h5")
        else:
            raise ValueError(f"Unknown split: {split}")

        self.files = sorted(glob.glob(patt))
        if len(self.files) == 0:
            raise RuntimeError(f"No H5 files found at: {patt}")
        print(f"  Split {split}: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with h5py.File(self.files[i], "r") as f:
            rgb = f["rgb"][:]       # (3, 480, 640) uint8
            depth = f["depth"][:]   # (480, 640) float32
        img = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
        return NYUMatDataset._process(self, img, depth)


def make_datasets(args):
    """Auto-detect data format and create train/val datasets."""
    if args.data_format == "auto":
        if args.mat_path and os.path.isfile(args.mat_path):
            fmt = "mat"
        elif args.h5_root and os.path.isdir(args.h5_root):
            fmt = "h5"
        else:
            raise RuntimeError(
                f"Could not auto-detect data format. "
                f"Provide --mat_path (file: {args.mat_path}) "
                f"or --h5_root (dir: {args.h5_root})")
    else:
        fmt = args.data_format

    if fmt == "mat":
        print(f"Data format: .mat ({args.mat_path})")
        train_ds = NYUMatDataset(args.mat_path, args.train_split,
                                 args.image_size, augment=True)
        val_ds   = NYUMatDataset(args.mat_path, args.test_split,
                                 args.image_size, augment=False)
    else:
        print(f"Data format: H5 ({args.h5_root})")
        train_ds = NYUH5Dataset(args.h5_root, split="train",
                                image_size=args.image_size, augment=True)
        val_ds   = NYUH5Dataset(args.h5_root, split="val",
                                image_size=args.image_size, augment=False)
    return train_ds, val_ds


# ============================================================================
# Model: Official DINOv2 BNHead for depth
#
# Architecture (dinov2/hub/depth/decode_heads.py BNHead):
#   1. get_intermediate_layers(n=indices, reshape=True,
#                              return_class_token=True, norm=False)
#      → list of ([B,D,H_p,W_p], [B,D]) tuples
#   2. Per layer: concat patches + broadcast CLS → [B, 2D, H_p, W_p]
#   3. Upsample each to (H_p*4, W_p*4) then concat → [B, 4*2D, H', W']
#   4. Conv2d(8D → 256, kernel_size=1)
#   5. bins_strategy="UD": linspace(min_depth, max_depth, 256)
#   6. norm_strategy="linear": ReLU + 0.1 eps + L1-normalize
#   7. depth = einsum(probs, bin_centers)
#   8. Bilinear upsample to input resolution, clamp
# ============================================================================

class OfficialBNHead(nn.Module):
    def __init__(self, backbone, embed_dim, layer_indices,
                 n_bins=NUM_BINS, patch_size=14,
                 min_depth=DEPTH_MIN, max_depth=DEPTH_MAX,
                 upsample=HEAD_UPSAMPLE):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.layer_indices = list(layer_indices)
        self.n_layers = len(self.layer_indices)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.upsample = upsample

        # channels = n_layers × (patches + cls) × embed_dim = n_layers × 2 × D
        in_channels = self.n_layers * 2 * embed_dim
        self.conv_depth = nn.Conv2d(in_channels, n_bins,
                                    kernel_size=1, padding=0, stride=1)

        # UD bins: linear spacing (not log)
        bin_centers = torch.linspace(min_depth, max_depth, n_bins)
        self.register_buffer("bin_centers", bin_centers)

    def _build_features(self, x):
        """Extract and concatenate multi-layer features with CLS tokens."""
        B, C, H, W = x.shape
        H_p, W_p = H // self.patch_size, W // self.patch_size
        target_size = (H_p * self.upsample, W_p * self.upsample)

        with torch.no_grad():
            # Official: reshape=True, return_class_token=True, norm=False
            feats = self.backbone.get_intermediate_layers(
                x, n=self.layer_indices,
                reshape=True,
                return_class_token=True,
                norm=False,
            )

        per_layer = []
        for patches, cls_token in feats:
            # patches: [B, D, H_p, W_p], cls_token: [B, D]
            cls_b = cls_token[:, :, None, None].expand_as(patches)
            cat = torch.cat([patches, cls_b], dim=1)  # [B, 2D, H_p, W_p]
            cat_up = F.interpolate(cat, size=target_size,
                                   mode="bilinear", align_corners=False)
            per_layer.append(cat_up)
        return torch.cat(per_layer, dim=1)  # [B, n_layers*2D, H', W']

    def depth_pred(self, feat):
        """BNHead depth prediction: classify + UD bins + linear norm."""
        logit = self.conv_depth(feat)              # [B, K, H, W]

        # norm_strategy="linear": ReLU → +eps → L1 normalize (following AdaBins)
        logit = F.relu(logit) + NORM_EPS
        logit = logit / logit.sum(dim=1, keepdim=True)

        # Linear combination with bin centers
        depth = torch.einsum("bkhw,k->bhw", logit, self.bin_centers)
        return depth

    def forward(self, x):
        """Full forward: features → depth at input resolution."""
        B, C, H, W = x.shape
        feat = self._build_features(x)
        depth_lr = self.depth_pred(feat)                  # [B, H', W']
        depth_hr = F.interpolate(depth_lr.unsqueeze(1), size=(H, W),
                                  mode="bilinear", align_corners=False)
        depth_hr = depth_hr.squeeze(1).clamp(min=self.min_depth,
                                              max=self.max_depth)
        return depth_hr  # [B, H, W]


# ============================================================================
# Loss: SigLoss (official DINOv2 / AdaBins / Eigen 2014)
#
# Source: dinov2/eval/depth/models/losses/sigloss.py
#   g = log(pred + eps) - log(gt + eps)
#   L = loss_weight * sqrt(var(g) + 0.15 * mean(g)^2)
# ============================================================================

class SigLoss(nn.Module):
    """Port of DINOv2 SigLoss with warm-up support."""
    def __init__(self, loss_weight=10.0, eps=SI_EPS,
                 warm_up=True, warm_iter=100):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def forward(self, pred, target, mask):
        if not mask.any():
            return torch.zeros((), device=pred.device, requires_grad=True)

        g = (torch.log(pred[mask] + self.eps)
             - torch.log(target[mask] + self.eps))

        if self.warm_up and self.warm_up_counter < self.warm_iter:
            # Warm-up: only use mean^2 term (gentler gradient)
            Dg = 0.15 * torch.pow(torch.mean(g), 2)
            self.warm_up_counter += 1
        else:
            # Full loss: var(g) + 0.15 * mean(g)^2
            Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)

        return self.loss_weight * torch.sqrt(Dg.clamp(min=1e-8))


# ============================================================================
# Eval
# ============================================================================

@torch.no_grad()
def evaluate_rmse(model, val_loader, device, verbose=False):
    model.eval()
    sq_sum, count = 0.0, 0
    pred_sum, gt_sum = 0.0, 0.0
    pred_min, pred_max = float("inf"), float("-inf")
    for img, depth, mask in tqdm(val_loader, desc="eval",
                                  leave=False, dynamic_ncols=True):
        img = img.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pred_d = model(img)
        err = (pred_d - depth) ** 2
        sq_sum += err[mask].sum().item()
        count += mask.sum().item()
        pred_sum += pred_d[mask].sum().item()
        gt_sum   += depth[mask].sum().item()
        pred_min = min(pred_min, pred_d[mask].min().item())
        pred_max = max(pred_max, pred_d[mask].max().item())
    rmse = (sq_sum / max(count, 1)) ** 0.5
    if verbose:
        print(f"    [diag] pred mean={pred_sum/max(count,1):.3f}m, "
              f"gt mean={gt_sum/max(count,1):.3f}m, "
              f"pred range=[{pred_min:.3f}, {pred_max:.3f}]m")
    return rmse


# ============================================================================
# Train
# ============================================================================

def train_one(model_name, args, device):
    print(f"\n{'='*68}")
    print(f"Training (Official BNHead): {model_name} (seed={args.seed})")
    print('=' * 68)

    backbone = torch.hub.load("facebookresearch/dinov2", model_name)
    backbone = backbone.to(device).eval()
    embed_dim = backbone.embed_dim
    patch_size = backbone.patch_size

    # Layer indices
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
    print(f"  head input channels = {len(layer_indices) * 2 * embed_dim} "
          f"({len(layer_indices)} layers x 2 x {embed_dim}, includes CLS)")
    print(f"  n_bins={NUM_BINS}, bins=UD(linspace), "
          f"norm=linear(ReLU+eps+L1), upsample={HEAD_UPSAMPLE}x")

    model = OfficialBNHead(
        backbone, embed_dim, layer_indices,
        n_bins=NUM_BINS, patch_size=patch_size,
        upsample=HEAD_UPSAMPLE,
    ).to(device)

    train_ds, val_ds = make_datasets(args)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Loss: official SigLoss with warm-up
    criterion = SigLoss(loss_weight=args.loss_weight, warm_up=True, warm_iter=100)

    # Optimizer: AdamW with weight_decay=0.01 (official setting)
    optimizer = torch.optim.AdamW(model.conv_depth.parameters(),
                                   lr=args.lr, weight_decay=0.01,
                                   betas=(0.9, 0.999))

    # LR schedule: 30% linear warm-up → cosine annealing
    warmup_iters = int(0.3 * args.n_iter)
    def lr_lambda(step):
        if step < warmup_iters:
            return float(step + 1) / float(warmup_iters)
        progress = (step - warmup_iters) / max(1, args.n_iter - warmup_iters)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    model.conv_depth.train()
    it = iter(train_loader)
    t0 = time.time()
    pbar = tqdm(range(args.n_iter), desc="train", dynamic_ncols=True)
    rl_loss, rl_pred = None, None

    for step in pbar:
        try:
            img, depth, mask = next(it)
        except StopIteration:
            it = iter(train_loader)
            img, depth, mask = next(it)

        img = img.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        pred_d = model(img)
        loss = criterion(pred_d, depth, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_v = loss.item()
        pred_mean_v = pred_d[mask].mean().item() if mask.any() else 0.0
        rl_loss = loss_v      if rl_loss is None else 0.99 * rl_loss + 0.01 * loss_v
        rl_pred = pred_mean_v if rl_pred is None else 0.99 * rl_pred + 0.01 * pred_mean_v
        if step % 50 == 0:
            pbar.set_postfix({
                "loss": f"{rl_loss:.3f}",
                "pred_m": f"{rl_pred:.2f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        if step > 0 and step % args.eval_every == 0 and step < args.n_iter - 1:
            rmse = evaluate_rmse(model, val_loader, device, verbose=True)
            tqdm.write(f"  [step {step}/{args.n_iter}] RMSE = {rmse:.4f} m")
            model.conv_depth.train()

    final_rmse = evaluate_rmse(model, val_loader, device, verbose=True)
    elapsed = (time.time() - t0) / 60
    print(f"\n  -> {model_name} (seed={args.seed}) final RMSE = "
          f"{final_rmse:.4f} m   ({elapsed:.1f} min)")

    del backbone, model
    torch.cuda.empty_cache()
    return final_rmse


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    # Data: .mat format
    p.add_argument("--mat_path", default="/nas/datahub/nyud/nyu_depth_v2_labeled.mat")
    p.add_argument("--train_split", default="/nas/datahub/nyud/train.txt")
    p.add_argument("--test_split", default="/nas/datahub/nyud/test.txt")
    # Data: HF H5 format
    p.add_argument("--h5_root", default="/data/ysk1207/nyu_depth_v2/data/extracted")
    # Auto-detect format (prioritize .mat if both available)
    p.add_argument("--data_format", default="auto",
                   choices=["auto", "mat", "h5"],
                   help="Data format: auto (detect), mat (.mat+split), h5 (HF H5)")
    # Model
    p.add_argument("--models", nargs="+",
                   default=["dinov2_vitl14", "dinov2_vitl14_reg"])
    p.add_argument("--layer_indices", type=int, nargs="+", default=None,
                   help="Override 0-indexed block indices. "
                        "Default: auto (ViT-L -> [4,11,17,23])")
    # Training
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Official: 1e-4 with AdamW")
    p.add_argument("--n_iter", type=int, default=38400,
                   help="Paper: 38.4k iterations")
    p.add_argument("--eval_every", type=int, default=2500)
    p.add_argument("--loss_weight", type=float, default=10.0,
                   help="SigLoss weight (official: 10.0)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", default="./results_phase3_v2.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"Device: {device}, batch_size={args.batch_size}, n_iter={args.n_iter}")
    print(f"image_size={args.image_size}, lr={args.lr}, loss_weight={args.loss_weight}")
    print(f"Layer indices: {args.layer_indices or 'auto (paper default)'}")

    results = {}
    for m in args.models:
        try:
            results[m] = round(train_one(m, args, device), 4)
        except Exception as e:
            print(f"  ERROR for {m}: {e}")
            import traceback; traceback.print_exc()
            results[m] = None
        with open(args.output, "w") as f:
            json.dump({"seed": args.seed, "results": results}, f, indent=2)

    print(f"\n{'='*64}")
    print(f"FINAL (seed={args.seed}) — NYUd Depth (Official BNHead)")
    print('=' * 64)
    for m, v in results.items():
        display = f"{v:.4f}" if v is not None else "FAILED"
        print(f"  {m:32}  {display:>10}")
    valid = [v for v in results.values() if v is not None]
    if len(valid) == 2:
        diff = valid[1] - valid[0]
        sign = "+" if diff >= 0 else ""
        print(f"\n  Delta (reg - no_reg) = {sign}{diff:.4f}")
        print(f"  Paper: 0.378 -> 0.366 (Delta = -0.012)")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
