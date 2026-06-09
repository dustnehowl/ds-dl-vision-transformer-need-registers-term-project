# siglip_artifact_probe_full.py

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from transformers import AutoImageProcessor, SiglipVisionModel


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def ensure_outdir(outdir: str) -> Path:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    return outdir_path


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def load_siglip_model(model_name: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def forward_siglip(
    image: Image.Image,
    processor,
    model: SiglipVisionModel,
    device: str,
):
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    outputs = model(
        pixel_values=pixel_values,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    )
    return outputs


def get_patch_features_from_hidden_state(
    hidden_state: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """
    SigLIP hidden_state shape:
        [batch, num_patches, hidden_dim]
    """
    feats = hidden_state[0]  # [seq_len, hidden_dim]
    seq_len = feats.shape[0]
    grid_size = int(math.sqrt(seq_len))

    if grid_size * grid_size != seq_len:
        raise ValueError(
            f"Cannot reshape seq_len={seq_len} into square grid. "
            f"sqrt(seq_len)={math.sqrt(seq_len):.3f}"
        )

    return feats, grid_size


def compute_norm_map_from_features(
    feats: torch.Tensor,
    grid_size: int,
) -> np.ndarray:
    norms = feats.norm(dim=-1)
    norm_map = norms.reshape(grid_size, grid_size)
    return tensor_to_numpy(norm_map)


def compute_all_layer_norm_maps(
    hidden_states: Tuple[torch.Tensor, ...],
) -> List[np.ndarray]:
    norm_maps = []

    for hs in hidden_states:
        feats, grid_size = get_patch_features_from_hidden_state(hs)
        norm_map = compute_norm_map_from_features(feats, grid_size)
        norm_maps.append(norm_map)

    return norm_maps


def get_outlier_mask(
    norm_map: np.ndarray,
    method: str = "iqr",
    percentile: float = 98.0,
    zscore: float = 3.0,
    iqr_scale: float = 3.0,
    manual_threshold: float = None,
):
    vals = norm_map.reshape(-1)

    if manual_threshold is not None:
        threshold = manual_threshold

    elif method == "iqr":
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        threshold = q3 + iqr_scale * iqr

    elif method == "percentile":
        threshold = np.percentile(vals, percentile)

    elif method == "zscore":
        threshold = vals.mean() + zscore * vals.std()

    else:
        raise ValueError(f"Unknown threshold method: {method}")

    mask = norm_map > threshold
    return mask, float(threshold)


def compute_layer_norm_stats(
    hidden_states: Tuple[torch.Tensor, ...],
) -> Dict[str, np.ndarray]:
    layer_ids = []
    mean_vals = []
    median_vals = []
    std_vals = []
    p95_vals = []
    p99_vals = []
    max_vals = []

    for layer_idx, hs in enumerate(hidden_states):
        feats, _ = get_patch_features_from_hidden_state(hs)
        norms = feats.norm(dim=-1)
        norms_np = tensor_to_numpy(norms)

        layer_ids.append(layer_idx)
        mean_vals.append(norms_np.mean())
        median_vals.append(np.median(norms_np))
        std_vals.append(norms_np.std())
        p95_vals.append(np.percentile(norms_np, 95))
        p99_vals.append(np.percentile(norms_np, 99))
        max_vals.append(norms_np.max())

    return {
        "layer": np.array(layer_ids),
        "mean": np.array(mean_vals),
        "median": np.array(median_vals),
        "std": np.array(std_vals),
        "p95": np.array(p95_vals),
        "p99": np.array(p99_vals),
        "max": np.array(max_vals),
    }


def save_final_layer_summary(
    image: Image.Image,
    norm_map: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    save_path: Path,
    title: str,
):
    outlier_ratio = mask.mean() * 100.0

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    im1 = axes[1].imshow(norm_map, cmap="magma")
    axes[1].set_title("Final layer patch-token L2 norm")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    vals = norm_map.reshape(-1)
    axes[2].hist(vals, bins=50)
    axes[2].axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"threshold={threshold:.2f}",
    )
    axes[2].set_title("Final layer norm histogram")
    axes[2].set_xlabel("L2 norm")
    axes[2].set_ylabel("Patch count")
    axes[2].legend()

    small_img = image.resize((norm_map.shape[1], norm_map.shape[0]))
    axes[3].imshow(small_img)
    axes[3].imshow(mask, cmap="Reds", alpha=0.55)
    axes[3].set_title(f"Outlier overlay ({outlier_ratio:.2f}%)")
    axes[3].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_layer_norm_curve(
    stats: Dict[str, np.ndarray],
    save_path: Path,
    title: str,
):
    plt.figure(figsize=(9, 5.5))

    plt.plot(stats["layer"], stats["mean"], marker="o", label="mean")
    plt.plot(stats["layer"], stats["median"], marker="o", label="median")
    plt.plot(stats["layer"], stats["p95"], marker="o", label="p95")
    plt.plot(stats["layer"], stats["p99"], marker="o", label="p99")
    plt.plot(stats["layer"], stats["max"], marker="o", label="max")

    plt.xlabel("Layer index")
    plt.ylabel("Patch-token L2 norm")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_layer_hist_grid(
    hidden_states: Tuple[torch.Tensor, ...],
    save_path: Path,
    title: str,
    num_layers_to_show: int = 8,
):
    total_layers = len(hidden_states)
    selected_layers = np.linspace(0, total_layers - 1, num_layers_to_show, dtype=int)

    cols = 4
    rows = int(np.ceil(num_layers_to_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax_idx, layer_idx in enumerate(selected_layers):
        hs = hidden_states[layer_idx]
        feats, _ = get_patch_features_from_hidden_state(hs)
        norms = tensor_to_numpy(feats.norm(dim=-1))

        axes[ax_idx].hist(norms, bins=40)
        axes[ax_idx].set_title(
            f"Layer {layer_idx}\nmean={norms.mean():.2f}, max={norms.max():.2f}"
        )
        axes[ax_idx].set_xlabel("L2 norm")
        axes[ax_idx].set_ylabel("count")

    for j in range(len(selected_layers), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_layer_norm_map_grid(
    norm_maps: List[np.ndarray],
    save_path: Path,
    title: str,
    num_layers_to_show: int = 8,
    shared_scale: bool = True,
):
    total_layers = len(norm_maps)
    selected_layers = np.linspace(0, total_layers - 1, num_layers_to_show, dtype=int)
    selected_maps = [norm_maps[i] for i in selected_layers]

    if shared_scale:
        vmin = min(m.min() for m in selected_maps)
        vmax = max(m.max() for m in selected_maps)
    else:
        vmin = None
        vmax = None

    cols = 4
    rows = int(np.ceil(num_layers_to_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows))
    axes = np.array(axes).reshape(-1)

    for ax_idx, layer_idx in enumerate(selected_layers):
        norm_map = norm_maps[layer_idx]
        im = axes[ax_idx].imshow(norm_map, cmap="magma", vmin=vmin, vmax=vmax)
        axes[ax_idx].set_title(
            f"Layer {layer_idx}\nmean={norm_map.mean():.2f}, max={norm_map.max():.2f}"
        )
        axes[ax_idx].axis("off")
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)

    for j in range(len(selected_layers), len(axes)):
        axes[j].axis("off")

    scale_msg = "shared color scale" if shared_scale else "independent color scale"
    fig.suptitle(f"{title} ({scale_msg})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_stats_csv(
    stats: Dict[str, np.ndarray],
    save_path: Path,
):
    header = "layer,mean,median,std,p95,p99,max\n"
    lines = [header]

    for i in range(len(stats["layer"])):
        lines.append(
            f"{int(stats['layer'][i])},"
            f"{stats['mean'][i]:.6f},"
            f"{stats['median'][i]:.6f},"
            f"{stats['std'][i]:.6f},"
            f"{stats['p95'][i]:.6f},"
            f"{stats['p99'][i]:.6f},"
            f"{stats['max'][i]:.6f}\n"
        )

    save_path.write_text("".join(lines), encoding="utf-8")


def safe_filename_model(model_name: str) -> str:
    return model_name.replace("/", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize high-norm patch-token artifacts in one SigLIP image."
    )

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--outdir", type=str, default="./outputs_siglip_visual_one")

    parser.add_argument(
        "--manual_threshold",
        type=float,
        default=None,
        help="Manual threshold for outlier mask. Use 150 for your current SigLIP experiment.",
    )

    parser.add_argument(
        "--threshold_method",
        type=str,
        default="iqr",
        choices=["iqr", "percentile", "zscore"],
    )
    parser.add_argument("--percentile", type=float, default=98.0)
    parser.add_argument("--zscore", type=float, default=3.0)
    parser.add_argument("--iqr_scale", type=float, default=3.0)

    parser.add_argument("--num_layers_to_show", type=int, default=8)
    parser.add_argument(
        "--no_shared_scale",
        action="store_true",
        help="Use independent color scale for each layer norm map.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_path = Path(args.image)
    image_stem = image_path.stem
    outdir = ensure_outdir(args.outdir)
    safe_model_name = safe_filename_model(args.model)

    print("=" * 80)
    print("SigLIP single-image artifact probe")
    print("=" * 80)
    print(f"image: {args.image}")
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"outdir: {outdir}")
    print(f"manual_threshold: {args.manual_threshold}")
    print("=" * 80)

    image = load_image(args.image)
    processor, model = load_siglip_model(args.model, device)

    outputs = forward_siglip(
        image=image,
        processor=processor,
        model=model,
        device=device,
    )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("hidden_states is None. Check output_hidden_states=True.")

    norm_maps = compute_all_layer_norm_maps(hidden_states)
    final_norm_map = norm_maps[-1]

    final_mask, final_threshold = get_outlier_mask(
        final_norm_map,
        method=args.threshold_method,
        percentile=args.percentile,
        zscore=args.zscore,
        iqr_scale=args.iqr_scale,
        manual_threshold=args.manual_threshold,
    )

    final_outlier_ratio = final_mask.mean() * 100.0
    stats = compute_layer_norm_stats(hidden_states)

    final_summary_path = outdir / f"{image_stem}_{safe_model_name}_final_summary.png"
    curve_path = outdir / f"{image_stem}_{safe_model_name}_layer_norm_curve.png"
    hist_grid_path = outdir / f"{image_stem}_{safe_model_name}_layer_hist_grid.png"
    map_grid_path = outdir / f"{image_stem}_{safe_model_name}_layer_norm_map_grid.png"
    csv_path = outdir / f"{image_stem}_{safe_model_name}_layer_norm_stats.csv"

    title_base = f"{args.model} | {image_path.name}"

    save_final_layer_summary(
        image=image,
        norm_map=final_norm_map,
        mask=final_mask,
        threshold=final_threshold,
        save_path=final_summary_path,
        title=(
            f"{title_base}\n"
            f"Final layer norm map | threshold={final_threshold:.2f} | "
            f"outlier={final_outlier_ratio:.2f}%"
        ),
    )

    save_layer_norm_curve(
        stats=stats,
        save_path=curve_path,
        title=f"{title_base}\nLayer-wise patch-token norm statistics",
    )

    save_layer_hist_grid(
        hidden_states=hidden_states,
        save_path=hist_grid_path,
        title=f"{title_base}\nLayer-wise norm histograms",
        num_layers_to_show=args.num_layers_to_show,
    )

    save_layer_norm_map_grid(
        norm_maps=norm_maps,
        save_path=map_grid_path,
        title=f"{title_base}\nLayer-wise norm maps",
        num_layers_to_show=args.num_layers_to_show,
        shared_scale=not args.no_shared_scale,
    )

    save_stats_csv(stats, csv_path)

    print("[DONE]")
    print(f"final summary: {final_summary_path}")
    print(f"layer norm curve: {curve_path}")
    print(f"layer hist grid: {hist_grid_path}")
    print(f"layer norm map grid: {map_grid_path}")
    print(f"csv stats: {csv_path}")
    print("-" * 80)
    print("Final layer statistics")
    print(f"grid size: {final_norm_map.shape[0]} x {final_norm_map.shape[1]}")
    print(f"min norm: {final_norm_map.min():.6f}")
    print(f"mean norm: {final_norm_map.mean():.6f}")
    print(f"median norm: {np.median(final_norm_map):.6f}")
    print(f"p95 norm: {np.percentile(final_norm_map, 95):.6f}")
    print(f"p99 norm: {np.percentile(final_norm_map, 99):.6f}")
    print(f"max norm: {final_norm_map.max():.6f}")
    print(f"threshold: {final_threshold:.6f}")
    print(f"outlier ratio: {final_outlier_ratio:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
