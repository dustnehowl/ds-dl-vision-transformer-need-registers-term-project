import argparse
import os
import time
from typing import Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def build_imagenet_val_loader(data_dir: str, batch_size: int, num_images: int, num_workers: int):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    if num_images > 0:
        num_images = min(num_images, len(dataset))
        dataset = Subset(dataset, list(range(num_images)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader


def get_patch_grid_size(num_patches: int) -> Tuple[int, int]:
    h = int(num_patches ** 0.5)
    w = h
    if h * w != num_patches:
        raise ValueError(f"num_patches={num_patches} is not square.")
    return h, w


@torch.no_grad()
def compute_batch_cos(
    model,
    images: torch.Tensor,
    threshold: float,
    use_quantile: bool,
    quantile: float,
):
    """
    논문 Figure 5(a) 방식:
    - outlier 판단: final output patch token norm
    - cosine 계산: patch embedding layer 직후 token
    """

    # 1. patch embedding 직후 token
    patch_embed = model.patch_embed(images)  # [B, N, D]
    patch_embed = F.normalize(patch_embed, dim=-1)

    # 2. final output patch token
    feats = model.forward_features(images)
    # x_prenorm: [B, 1 + num_register_tokens + N, D]
    x_prenorm = feats["x_prenorm"]

    num_register_tokens = getattr(model, "num_register_tokens", 0)
    patch_start = 1 + num_register_tokens

    patch_out = x_prenorm[:, patch_start:, :]   # [B, N, D]
    patch_norm = patch_out.norm(dim=-1)         # [B, N]

    if use_quantile:
        local_threshold = torch.quantile(patch_norm.flatten(), quantile)
    else:
        local_threshold = torch.tensor(threshold, device=patch_norm.device)

    outlier_mask = patch_norm > local_threshold
    normal_mask = ~outlier_mask

    b, n, d = patch_embed.shape
    h, w = get_patch_grid_size(n)

    patch_grid = patch_embed.reshape(b, h, w, d)
    outlier_grid = outlier_mask.reshape(b, h, w)
    normal_grid = normal_mask.reshape(b, h, w)

    outlier_cos_values = []
    normal_cos_values = []

    # 상하좌우 neighbor pair
    directions = [
        (0, 1),   # right
        (0, -1),  # left
        (1, 0),   # down
        (-1, 0),  # up
    ]

    for dy, dx in directions:
        if dy == 0 and dx == 1:
            center = patch_grid[:, :, :-1, :]
            neighbor = patch_grid[:, :, 1:, :]
            out_m = outlier_grid[:, :, :-1]
            nor_m = normal_grid[:, :, :-1]
        elif dy == 0 and dx == -1:
            center = patch_grid[:, :, 1:, :]
            neighbor = patch_grid[:, :, :-1, :]
            out_m = outlier_grid[:, :, 1:]
            nor_m = normal_grid[:, :, 1:]
        elif dy == 1 and dx == 0:
            center = patch_grid[:, :-1, :, :]
            neighbor = patch_grid[:, 1:, :, :]
            out_m = outlier_grid[:, :-1, :]
            nor_m = normal_grid[:, :-1, :]
        elif dy == -1 and dx == 0:
            center = patch_grid[:, 1:, :, :]
            neighbor = patch_grid[:, :-1, :, :]
            out_m = outlier_grid[:, 1:, :]
            nor_m = normal_grid[:, 1:, :]
        else:
            raise RuntimeError("Invalid direction")

        cos = (center * neighbor).sum(dim=-1)  # 이미 normalize 했으므로 dot = cosine

        if out_m.any():
            outlier_cos_values.append(cos[out_m].detach().cpu())
        if nor_m.any():
            normal_cos_values.append(cos[nor_m].detach().cpu())

    if len(outlier_cos_values) > 0:
        outlier_cos_values = torch.cat(outlier_cos_values).numpy()
    else:
        outlier_cos_values = np.array([], dtype=np.float32)

    if len(normal_cos_values) > 0:
        normal_cos_values = torch.cat(normal_cos_values).numpy()
    else:
        normal_cos_values = np.array([], dtype=np.float32)

    return normal_cos_values, outlier_cos_values, patch_norm.detach().cpu().numpy(), float(local_threshold.item())


def plot_density(normal_cos, outlier_cos, output_png, title=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(3.0, 2.4))

    sns.kdeplot(
        normal_cos,
        label="normal patches",
        color="#4C72B0",
        linewidth=1.5,
        fill=False,
        bw_adjust=0.8,
        clip=(-0.1, 1.02),
    )

    sns.kdeplot(
        outlier_cos,
        label="artifact patches",
        color="#DD8452",
        linewidth=1.5,
        fill=False,
        bw_adjust=0.8,
        clip=(-0.1, 1.02),
    )

    plt.xlabel("")
    plt.ylabel("density")

    plt.xlim(-0.1, 1.02)
    plt.ylim(0, 20)

    plt.xticks([0.0, 0.5, 1.0])
    plt.yticks([0, 10, 20])

    plt.legend(
        loc="upper left",
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"[Saved plot] {output_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/imagenet_1k/val")
    parser.add_argument("--model", type=str, default="dinov2_vitb14")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=150.0)
    parser.add_argument("--use_quantile", action="store_true")
    parser.add_argument("--quantile", type=float, default=0.98)
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Model] {args.model}")
    print(f"[Data] {args.data_dir}")
    print(f"[Num images] {args.num_images}")
    print(f"[Batch size] {args.batch_size}")

    loader = build_imagenet_val_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_images=args.num_images,
        num_workers=args.num_workers,
    )

    model = torch.hub.load("facebookresearch/dinov2", args.model)
    model.eval().to(device)

    all_normal_cos = []
    all_outlier_cos = []
    all_norms = []
    thresholds = []

    start = time.time()

    for images, _ in tqdm(loader):
        images = images.to(device, non_blocking=True)

        normal_cos, outlier_cos, patch_norms, used_threshold = compute_batch_cos(
            model=model,
            images=images,
            threshold=args.threshold,
            use_quantile=args.use_quantile,
            quantile=args.quantile,
        )

        if len(normal_cos) > 0:
            all_normal_cos.append(normal_cos)
        if len(outlier_cos) > 0:
            all_outlier_cos.append(outlier_cos)

        all_norms.append(patch_norms.reshape(-1))
        thresholds.append(used_threshold)

    all_normal_cos = np.concatenate(all_normal_cos) if all_normal_cos else np.array([])
    all_outlier_cos = np.concatenate(all_outlier_cos) if all_outlier_cos else np.array([])
    all_norms = np.concatenate(all_norms) if all_norms else np.array([])

    elapsed = time.time() - start

    print("\n===== Result Summary =====")
    print(f"elapsed sec: {elapsed:.2f}")
    print(f"normal cos count: {len(all_normal_cos)}")
    print(f"outlier cos count: {len(all_outlier_cos)}")
    print(f"norm min / mean / max: {all_norms.min():.3f} / {all_norms.mean():.3f} / {all_norms.max():.3f}")
    print(f"threshold mean: {np.mean(thresholds):.3f}")

    if len(all_normal_cos) > 0:
        print(f"normal cos mean: {all_normal_cos.mean():.4f}")
    if len(all_outlier_cos) > 0:
        print(f"outlier cos mean: {all_outlier_cos.mean():.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    tag = f"{args.model}_n{args.num_images}"
    if args.use_quantile:
        tag += f"_q{args.quantile}"
    else:
        tag += f"_thr{args.threshold}"

    tag = f"{tag}_{timestamp}"

    npz_path = os.path.join(args.output_dir, f"cos_values_{tag}.npz")
    png_path = os.path.join(args.output_dir, f"cos_density_{tag}.png")

    np.savez(
        npz_path,
        normal_cos=all_normal_cos,
        outlier_cos=all_outlier_cos,
        patch_norms=all_norms,
        thresholds=np.array(thresholds),
        args=vars(args),
    )
    print(f"[Saved npz] {npz_path}")

    title = f"{args.model} | ImageNet val | n={args.num_images}"
    plot_density(all_normal_cos, all_outlier_cos, png_path, title)


if __name__ == "__main__":
    main()
