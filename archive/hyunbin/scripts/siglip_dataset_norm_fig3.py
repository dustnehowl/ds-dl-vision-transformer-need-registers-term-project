# siglip_dataset_norm_fig3.py

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from transformers import AutoImageProcessor, SiglipVisionModel


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_image_paths(image_dir: str):
    image_dir = Path(image_dir)
    return sorted([
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def load_image(path: Path):
    return Image.open(path).convert("RGB")


@torch.no_grad()
def extract_final_patch_norms(image, processor, model, device):
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    outputs = model(
        pixel_values=pixel_values,
        output_hidden_states=True,
        return_dict=True,
    )

    final_hidden = outputs.hidden_states[-1][0]  # [num_patches, hidden_dim]

    seq_len = final_hidden.shape[0]
    grid_size = int(math.sqrt(seq_len))

    if grid_size * grid_size != seq_len:
        raise ValueError(f"Sequence length {seq_len} cannot be reshaped to square grid.")

    norms = final_hidden.norm(dim=-1)
    return norms.detach().cpu().float().numpy(), grid_size


def init_csv(csv_path: Path):
    if csv_path.exists():
        return

    fieldnames = [
        "image_path",
        "grid_size",
        "num_patches",
        "mean",
        "median",
        "std",
        "p90",
        "p95",
        "p99",
        "p99_5",
        "p99_9",
        "max",
        "artifact_score_max_over_median",
        "tail_score_p99_over_median",
        "tail_score_p99_9_over_median",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def load_processed_paths(csv_path: Path):
    processed = set()

    if not csv_path.exists():
        return processed

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["image_path"])

    return processed


def append_row(csv_path: Path, row: dict):
    fieldnames = [
        "image_path",
        "grid_size",
        "num_patches",
        "mean",
        "median",
        "std",
        "p90",
        "p95",
        "p99",
        "p99_5",
        "p99_9",
        "max",
        "artifact_score_max_over_median",
        "tail_score_p99_over_median",
        "tail_score_p99_9_over_median",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def update_hist_counts(norms, hist_counts, bin_edges):
    counts, _ = np.histogram(norms, bins=bin_edges)
    hist_counts += counts
    return hist_counts


def estimate_percentile_from_hist(counts, bin_edges, percentile):
    cumulative = np.cumsum(counts)
    total = cumulative[-1]

    if total == 0:
        return np.nan

    target = total * percentile / 100.0
    idx = np.searchsorted(cumulative, target)
    idx = min(idx, len(bin_edges) - 2)

    return float(bin_edges[idx])


def estimate_mean_from_hist(counts, bin_edges):
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = counts.sum()

    if total == 0:
        return np.nan

    return float((counts * centers).sum() / total)


def estimate_outlier_count_from_hist(counts, bin_edges, threshold):
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return int(counts[centers > threshold].sum())


def save_paper_style_histogram(
    counts,
    bin_edges,
    save_path: Path,
    title: str,
    threshold=None,
    xlim=(0, 600),
    ylim=(1e-5, 1e-1),
    show_threshold=True,
):
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    widths = np.diff(bin_edges)

    total = counts.sum()
    density = counts / (total * widths + 1e-12)

    mean = estimate_mean_from_hist(counts, bin_edges)
    median = estimate_percentile_from_hist(counts, bin_edges, 50)
    p99 = estimate_percentile_from_hist(counts, bin_edges, 99)
    p99_9 = estimate_percentile_from_hist(counts, bin_edges, 99.9)

    if threshold is None:
        threshold = p99_9

    outlier_count = estimate_outlier_count_from_hist(counts, bin_edges, threshold)
    outlier_ratio = 100.0 * outlier_count / max(int(total), 1)

    fig, ax = plt.subplots(figsize=(3.0, 2.5))

    ax.bar(
        centers,
        density,
        width=widths,
        color="#1f77b4",
        align="center",
        linewidth=0,
    )

    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xticks([0, 200, 400, 600])
    ax.set_yticks([1e-5, 1e-3, 1e-1])

    ax.set_title(title, fontsize=18, fontfamily="serif")
    ax.set_xlabel(r"$L_2$ norm", fontsize=18, fontfamily="serif")

    if show_threshold:
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.5,
        )

    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.tick_params(axis="both", which="minor", direction="in")

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return {
        "total_patch_tokens": int(total),
        "mean_estimated_from_hist": mean,
        "median_estimated_from_hist": median,
        "p99_estimated_from_hist": p99,
        "p99_9_estimated_from_hist": p99_9,
        "threshold": float(threshold),
        "outlier_count": outlier_count,
        "outlier_ratio_percent": outlier_ratio,
    }


def save_regular_histogram(
    counts,
    bin_edges,
    save_path: Path,
    title: str,
    log_y=False,
):
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    widths = np.diff(bin_edges)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        centers,
        counts,
        width=widths,
        align="center",
        color="#1f77b4",
        linewidth=0,
    )

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Patch-token L2 norm")
    ax.set_ylabel("Count" if not log_y else "Count, log scale")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_summary_json(summary_path: Path, summary: dict):
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--outdir", type=str, default="./outputs_siglip_fig3")

    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=500)

    parser.add_argument("--hist_max", type=float, default=600.0)
    parser.add_argument("--hist_bins", type=int, default=600)

    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--no_threshold_line",
        action="store_true",
        help="Do not draw threshold line on paper-style figure.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    safe_model_name = args.model.replace("/", "_")

    csv_path = outdir / f"{safe_model_name}_per_image_stats.csv"
    hist_counts_path = outdir / f"{safe_model_name}_hist_counts.npy"
    bin_edges_path = outdir / f"{safe_model_name}_hist_bin_edges.npy"

    paper_fig_path = outdir / f"{safe_model_name}_paper_style_norm_hist.png"
    regular_fig_path = outdir / f"{safe_model_name}_norm_hist_count.png"
    regular_log_fig_path = outdir / f"{safe_model_name}_norm_hist_count_log.png"

    summary_path = outdir / f"{safe_model_name}_summary.json"
    error_log_path = outdir / f"{safe_model_name}_errors.txt"

    image_paths = collect_image_paths(args.image_dir)

    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]

    init_csv(csv_path)

    if args.resume:
        processed_paths = load_processed_paths(csv_path)
    else:
        processed_paths = set()

    image_paths_to_run = [
        p for p in image_paths
        if str(p) not in processed_paths
    ]

    bin_edges = np.linspace(0.0, args.hist_max, args.hist_bins + 1)

    if args.resume and hist_counts_path.exists() and bin_edges_path.exists():
        hist_counts = np.load(hist_counts_path)
        bin_edges = np.load(bin_edges_path)
    else:
        hist_counts = np.zeros(args.hist_bins, dtype=np.int64)

    print("=" * 80)
    print("SigLIP Figure 3-style norm experiment")
    print("=" * 80)
    print(f"image_dir: {args.image_dir}")
    print(f"num images found: {len(image_paths)}")
    print(f"remaining images: {len(image_paths_to_run)}")
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"outdir: {outdir}")
    print(f"hist range: 0 ~ {args.hist_max}, bins={args.hist_bins}")
    print("=" * 80)

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = SiglipVisionModel.from_pretrained(args.model).to(device)
    model.eval()

    processed_this_run = 0
    total_patches_this_run = 0
    global_max_this_run = 0.0

    for idx, img_path in enumerate(tqdm(image_paths_to_run), start=1):
        try:
            image = load_image(img_path)

            norms, grid_size = extract_final_patch_norms(
                image=image,
                processor=processor,
                model=model,
                device=device,
            )

            median = float(np.median(norms))
            mean = float(np.mean(norms))
            std = float(np.std(norms))
            p90 = float(np.percentile(norms, 90))
            p95 = float(np.percentile(norms, 95))
            p99 = float(np.percentile(norms, 99))
            p99_5 = float(np.percentile(norms, 99.5))
            p99_9 = float(np.percentile(norms, 99.9))
            max_norm = float(np.max(norms))

            row = {
                "image_path": str(img_path),
                "grid_size": grid_size,
                "num_patches": len(norms),
                "mean": mean,
                "median": median,
                "std": std,
                "p90": p90,
                "p95": p95,
                "p99": p99,
                "p99_5": p99_5,
                "p99_9": p99_9,
                "max": max_norm,
                "artifact_score_max_over_median": max_norm / (median + 1e-8),
                "tail_score_p99_over_median": p99 / (median + 1e-8),
                "tail_score_p99_9_over_median": p99_9 / (median + 1e-8),
            }

            append_row(csv_path, row)
            hist_counts = update_hist_counts(norms, hist_counts, bin_edges)

            processed_this_run += 1
            total_patches_this_run += len(norms)
            global_max_this_run = max(global_max_this_run, max_norm)

            if idx % args.save_every == 0:
                np.save(hist_counts_path, hist_counts)
                np.save(bin_edges_path, bin_edges)

        except Exception as e:
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"{img_path}\t{repr(e)}\n")
            continue

    np.save(hist_counts_path, hist_counts)
    np.save(bin_edges_path, bin_edges)

    paper_stats = save_paper_style_histogram(
        counts=hist_counts,
        bin_edges=bin_edges,
        save_path=paper_fig_path,
        title="SigLIP norms",
        threshold=args.threshold,
        xlim=(0, 600),
        ylim=(1e-5, 1e-1),
        show_threshold=not args.no_threshold_line,
    )

    save_regular_histogram(
        counts=hist_counts,
        bin_edges=bin_edges,
        save_path=regular_fig_path,
        title=f"{args.model} final-layer patch-token norm histogram",
        log_y=False,
    )

    save_regular_histogram(
        counts=hist_counts,
        bin_edges=bin_edges,
        save_path=regular_log_fig_path,
        title=f"{args.model} final-layer patch-token norm histogram, log count",
        log_y=True,
    )

    summary = {
        "model": args.model,
        "image_dir": args.image_dir,
        "num_images_found": len(image_paths),
        "processed_this_run": processed_this_run,
        "total_patches_this_run": total_patches_this_run,
        "global_max_this_run": global_max_this_run,
        "hist_max": args.hist_max,
        "hist_bins": args.hist_bins,
        "paper_style_stats": paper_stats,
        "csv_path": str(csv_path),
        "hist_counts_path": str(hist_counts_path),
        "bin_edges_path": str(bin_edges_path),
        "paper_fig_path": str(paper_fig_path),
        "regular_fig_path": str(regular_fig_path),
        "regular_log_fig_path": str(regular_log_fig_path),
        "error_log_path": str(error_log_path),
    }

    save_summary_json(summary_path, summary)

    print("[DONE]")
    print(f"paper style figure: {paper_fig_path}")
    print(f"regular count figure: {regular_fig_path}")
    print(f"regular log figure: {regular_log_fig_path}")
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")
    print(f"errors: {error_log_path}")
    print(f"processed this run: {processed_this_run}")
    print(f"global max this run: {global_max_this_run}")
    print("paper style stats:")
    for k, v in paper_stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
