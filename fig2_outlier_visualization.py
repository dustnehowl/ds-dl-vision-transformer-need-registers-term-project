"""
=============================================================================
 DINOv2-g Outlier Token Visualization
 
 Goal: threshold = 150이 실제로 high-norm outlier patch를 잘 잡는지
       이미지와 함께 시각적으로 검증
 
 본 실험 (Table 1 재현) 들어가기 전 sanity check 용도
 
 출력:
   - 이미지별 4-panel 상세 그림 (개별 PNG)
   - 전체 비교 grid (overview PNG)
   - 통계 (outlier 비율, 분포 등)
=============================================================================
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision import transforms, datasets


# ============================================================================
# Preprocessing
# ============================================================================

def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(
            int(image_size * 256 / 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def denormalize_for_display(tensor):
    """Reverse ImageNet normalization → HWC numpy in [0,1]"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# ============================================================================
# Model & extraction
# ============================================================================

@torch.no_grad()
def extract_patch_norms(model, image_tensor, device):
    """
    Single image → patch norms grid (from x_prenorm)
    
    Returns:
        norms_grid: [H_p, W_p] numpy array of norm values
    """
    out = model.forward_features(image_tensor.unsqueeze(0).to(device))
    if "x_prenorm" not in out:
        raise RuntimeError("x_prenorm not in forward_features output. "
                           "Update DINOv2 hub model.")
    tokens = out["x_prenorm"]              # [1, 1+P, D]
    patches = tokens[0, 1:]                # [P, D]
    norms = patches.norm(dim=-1).cpu().numpy()
    
    side = int(np.sqrt(norms.shape[0]))
    assert side * side == norms.shape[0], f"Non-square patch grid? {norms.shape}"
    return norms.reshape(side, side)


# ============================================================================
# Visualization
# ============================================================================

def visualize_one(image_display, norms_grid, threshold, patch_size,
                  save_path, seed=0, name=""):
    """4-panel: original | norm heatmap | outlier mask | overlay"""
    rng = np.random.RandomState(seed)
    grid_h, grid_w = norms_grid.shape
    outlier_mask = norms_grid > threshold
    n_outliers = int(outlier_mask.sum())
    total = grid_h * grid_w
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.8))
    
    # 1. Original
    axes[0].imshow(image_display)
    axes[0].set_title(f"Original\n{name}", fontsize=11)
    axes[0].axis("off")
    
    # 2. Norm heatmap with threshold marker on colorbar
    vmax = max(norms_grid.max(), threshold * 1.1)
    im = axes[1].imshow(norms_grid, cmap="viridis", vmin=0, vmax=vmax)
    axes[1].set_title(
        f"Patch norms\nmin={norms_grid.min():.0f}, "
        f"max={norms_grid.max():.0f}, mean={norms_grid.mean():.0f}",
        fontsize=11,
    )
    axes[1].axis("off")
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.axhline(y=threshold, color="red", linewidth=2)
    cbar.ax.text(2.5, threshold, f" thr={threshold:.0f}",
                 color="red", va="center", fontsize=10,
                 transform=cbar.ax.get_yaxis_transform())
    
    # 3. Outlier binary mask
    axes[2].imshow(outlier_mask, cmap="Reds", vmin=0, vmax=1)
    pct = 100 * n_outliers / total
    axes[2].set_title(
        f"Outlier mask (norm > {threshold:.0f})\n"
        f"{n_outliers} / {total} patches ({pct:.1f}%)",
        fontsize=11,
    )
    axes[2].axis("off")
    
    # 4. Overlay on original
    axes[3].imshow(image_display)
    outlier_positions = []
    for i in range(grid_h):
        for j in range(grid_w):
            if outlier_mask[i, j]:
                outlier_positions.append((i, j))
                rect = Rectangle(
                    (j * patch_size, i * patch_size),
                    patch_size, patch_size,
                    linewidth=1.5, edgecolor="red",
                    facecolor="red", alpha=0.35,
                )
                axes[3].add_patch(rect)
    
    selected = None
    if outlier_positions:
        pick = rng.randint(len(outlier_positions))
        sel_i, sel_j = outlier_positions[pick]
        selected = (sel_i, sel_j)
        rect = Rectangle(
            (sel_j * patch_size, sel_i * patch_size),
            patch_size, patch_size,
            linewidth=3.5, edgecolor="yellow", facecolor="none",
        )
        axes[3].add_patch(rect)
        title4 = f"Outliers: red\nRandomly picked: yellow at ({sel_i},{sel_j})"
    else:
        title4 = "NO OUTLIERS FOUND\n(consider lowering threshold)"
    axes[3].set_title(title4, fontsize=11)
    axes[3].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close()
    
    return {
        "n_outliers": n_outliers,
        "total_patches": total,
        "max_norm": float(norms_grid.max()),
        "min_norm": float(norms_grid.min()),
        "mean_norm": float(norms_grid.mean()),
        "selected_pos": selected,
    }


def save_overview_grid(records, output_path, threshold):
    """All images in one grid for quick visual comparison"""
    n = len(records)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for k, rec in enumerate(records):
        r, c = k // cols, k % cols
        ax = axes[r, c]
        ax.imshow(rec["image_display"])
        
        norms_grid = rec["norms_grid"]
        patch_size = rec["patch_size"]
        outlier_mask = norms_grid > threshold
        H, W = norms_grid.shape
        for i in range(H):
            for j in range(W):
                if outlier_mask[i, j]:
                    rect = Rectangle(
                        (j * patch_size, i * patch_size),
                        patch_size, patch_size,
                        linewidth=1, edgecolor="red",
                        facecolor="red", alpha=0.45,
                    )
                    ax.add_patch(rect)
        
        n_out = int(outlier_mask.sum())
        ax.set_title(f"{rec['name'][:28]}\noutliers: {n_out}", fontsize=10)
        ax.axis("off")
    
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    
    fig.suptitle(f"Outlier overlay (threshold = {threshold})", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()


# ============================================================================
# Sample loading
# ============================================================================

def load_samples(args, transform):
    """Returns list of (tensor, display_name)"""
    samples = []
    
    if args.images_dir and os.path.isdir(args.images_dir):
        files = sorted([
            f for f in os.listdir(args.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ])[: args.num_images]
        for fname in files:
            img = Image.open(os.path.join(args.images_dir, fname)).convert("RGB")
            samples.append((transform(img), fname))
        if not samples:
            print(f"  WARNING: no images found in {args.images_dir}, "
                  f"falling back to dataset {args.dataset}")
        else:
            return samples
    
    # Fallback: torchvision dataset
    print(f"  Loading {args.num_images} random images from {args.dataset}...")
    if args.dataset == "Pets":
        ds = datasets.OxfordIIITPet(root=args.data_root, split="test",
                                     download=True, transform=transform)
    elif args.dataset == "Flowers102":
        ds = datasets.Flowers102(root=args.data_root, split="test",
                                  download=True, transform=transform)
    elif args.dataset == "Food101":
        ds = datasets.Food101(root=args.data_root, split="test",
                               download=True, transform=transform)
    else:
        raise ValueError(f"Unknown fallback dataset: {args.dataset}")
    
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(ds), size=min(args.num_images, len(ds)), replace=False)
    for idx in indices:
        tensor, _ = ds[idx]
        samples.append((tensor, f"{args.dataset}_idx{idx}"))
    return samples


# ============================================================================
# Main
# ============================================================================

def run(args):
    device = f"cuda:{args.gpu}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading {args.model}...")
    model = torch.hub.load("facebookresearch/dinov2", args.model).to(device).eval()
    patch_size = model.patch_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params={n_params:.0f}M, embed_dim={model.embed_dim}, "
          f"patch_size={patch_size}")
    
    side = args.image_size // patch_size
    total = side * side
    print(f"  image_size={args.image_size} → {side}x{side} = {total} patches/image")
    print(f"  threshold = {args.threshold}")
    print(f"  paper expected outlier ratio: ~2.37% "
          f"(≈ {total * 0.0237:.1f} outliers per image)\n")
    
    transform = get_transform(args.image_size)
    samples = load_samples(args, transform)
    print(f"Loaded {len(samples)} sample images\n")
    
    records = []
    for k, (tensor, name) in enumerate(samples):
        print(f"[{k+1}/{len(samples)}] {name}")
        norms_grid = extract_patch_norms(model, tensor, device)
        
        n_out = int((norms_grid > args.threshold).sum())
        print(f"  norm range: [{norms_grid.min():.1f}, {norms_grid.max():.1f}], "
              f"mean={norms_grid.mean():.1f}")
        print(f"  outliers: {n_out}/{total} ({100*n_out/total:.1f}%)")
        
        safe_name = "".join(c if c.isalnum() else "_" for c in name)[:40]
        save_path = os.path.join(args.output_dir, f"vis_{k:02d}_{safe_name}.png")
        image_display = denormalize_for_display(tensor)
        stats = visualize_one(image_display, norms_grid, args.threshold,
                              patch_size, save_path, seed=args.seed + k, name=name)
        print(f"  → {save_path}")
        
        records.append({
            "name": name,
            "image_display": image_display,
            "norms_grid": norms_grid,
            "patch_size": patch_size,
            **stats,
        })
        print()
    
    # Overview grid
    grid_path = os.path.join(args.output_dir, "_overview_grid.png")
    save_overview_grid(records, grid_path, args.threshold)
    print(f"Overview grid → {grid_path}\n")
    
    # Summary table
    print("=" * 76)
    print(f"Summary: {args.model}, image_size={args.image_size}, "
          f"threshold={args.threshold}")
    print("=" * 76)
    print(f"{'image':<42} {'n_out':>6} {'min':>6} {'max':>7} {'mean':>7}")
    print("-" * 76)
    for r in records:
        print(f"{r['name'][:40]:<42} {r['n_outliers']:>6d} "
              f"{r['min_norm']:>6.1f} {r['max_norm']:>7.1f} {r['mean_norm']:>7.1f}")
    
    total_out = sum(r["n_outliers"] for r in records)
    total_pat = total * len(records)
    n_with = sum(1 for r in records if r["n_outliers"] > 0)
    print()
    print(f"Overall outlier ratio: {total_out}/{total_pat} "
          f"({100*total_out/total_pat:.2f}%)   [paper: ~2.37%]")
    print(f"Images with ≥1 outlier: {n_with}/{len(records)} "
          f"({100*n_with/len(records):.0f}%)")
    
    # Sanity hints
    print("\nSanity checks:")
    if 100 * total_out / total_pat < 0.5:
        print("  ⚠ Outlier ratio is very low. Either:")
        print("     (a) threshold is too high → try --threshold 100 or lower")
        print("     (b) model has weak outliers → confirm you're using vitg14")
    elif 100 * total_out / total_pat > 10:
        print("  ⚠ Outlier ratio is unusually high → threshold may be too low")
    else:
        print("  ✓ Outlier ratio is in reasonable range")
    
    if n_with < len(records) * 0.5:
        print("  ⚠ More than half the images have no outliers at all")
        print("     → try higher resolution (--image_size 448) or lower threshold")
    else:
        print("  ✓ Most images have at least one outlier")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dinov2_vitg14",
                   choices=["dinov2_vits14", "dinov2_vitb14",
                            "dinov2_vitl14", "dinov2_vitg14"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--threshold", type=float, default=150.0,
                   help="Outlier norm threshold (paper: 150 for DINOv2-g)")
    p.add_argument("--num_images", type=int, default=8)
    
    # Source priority: images_dir > dataset
    p.add_argument("--images_dir", default=None,
                   help="Directory of your own images. If empty, use --dataset.")
    p.add_argument("--dataset", default="Pets",
                   choices=["Pets", "Flowers102", "Food101"])
    p.add_argument("--data_root", default="./data")
    
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="./outlier_vis")
    args = p.parse_args()
    
    run(args)