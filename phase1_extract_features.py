"""
=============================================================================
 Phase 1 Feature Extraction — Multi-GPU parallel

 ImageNet train/val에서 DINOv2 backbone feature를 미리 추출해 디스크에 저장.
 이후 phase1_train_linear.py에서 grid search만 수행.

 Usage:
   # 5 GPU 병렬 추출 (baseline + register)
   python phase1_extract_features.py --n_gpus 5

   # 특정 모델만
   python phase1_extract_features.py --n_gpus 5 --models dinov2_vitl14_reg
=============================================================================
"""

import os
import sys
import argparse
import subprocess
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_eval_transform(image_size=224):
    resize_size = int(image_size * 256 / 224)
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(resize_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.no_grad()
def extract_shard(model_name, split, shard_id, n_shards, args):
    """하나의 GPU에서 dataset shard의 feature를 추출."""
    device = f"cuda:{args.gpu}"
    print(f"[GPU {args.gpu}] {model_name} {split} shard {shard_id}/{n_shards}")

    # Load backbone
    backbone = torch.hub.load("facebookresearch/dinov2", model_name)
    backbone = backbone.to(device).eval()
    embed_dim = backbone.embed_dim
    n_last_blocks = 4  # max needed

    # Dataset
    split_dir = "train" if split == "train" else "val"
    ds = datasets.ImageFolder(
        os.path.join(args.imagenet_root, split_dir),
        transform=make_eval_transform(args.image_size),
    )

    # Shard indices
    total = len(ds)
    shard_size = (total + n_shards - 1) // n_shards
    start = shard_id * shard_size
    end = min(start + shard_size, total)
    indices = list(range(start, end))
    subset = Subset(ds, indices)
    print(f"  Shard [{start}:{end}] = {len(subset)} images")

    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Extract: last 4 blocks의 CLS tokens + last block의 mean patch tokens
    all_cls_tokens = []  # list of [B, 4, D]
    all_avg_patches = []  # list of [B, D]
    all_labels = []

    for images, labels in tqdm(loader, desc=f"shard{shard_id}",
                               leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        feats = backbone.get_intermediate_layers(
            images, n_last_blocks, return_class_token=True
        )
        # feats: list of (patch_tokens [B,P,D], cls_token [B,D]), length=4

        cls_tokens = torch.stack([cls for _, cls in feats], dim=1)  # [B, 4, D]
        avg_patch = feats[-1][0].mean(dim=1)  # [B, D] mean of last block patches

        all_cls_tokens.append(cls_tokens.cpu())
        all_avg_patches.append(avg_patch.cpu())
        all_labels.append(labels)

    result = {
        "cls_tokens": torch.cat(all_cls_tokens),      # [N, 4, D]
        "avg_patches": torch.cat(all_avg_patches),     # [N, D]
        "labels": torch.cat(all_labels),               # [N]
        "shard_id": shard_id,
        "start": start,
        "end": end,
    }

    # Save
    out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}_shard{shard_id}.pt")
    torch.save(result, out_path)
    print(f"  Saved: {out_path} "
          f"(cls={result['cls_tokens'].shape}, avg={result['avg_patches'].shape})")

    del backbone
    torch.cuda.empty_cache()


def merge_shards(model_name, split, n_shards, output_dir):
    """Shard 파일들을 하나로 합침."""
    feat_dir = os.path.join(output_dir, model_name)
    shards = []
    for i in range(n_shards):
        path = os.path.join(feat_dir, f"{split}_shard{i}.pt")
        if os.path.isfile(path):
            shards.append(torch.load(path, map_location="cpu", weights_only=True))

    if not shards:
        print(f"  WARNING: no shards found for {model_name}/{split}")
        return

    merged = {
        "cls_tokens": torch.cat([s["cls_tokens"] for s in shards]),
        "avg_patches": torch.cat([s["avg_patches"] for s in shards]),
        "labels": torch.cat([s["labels"] for s in shards]),
    }

    out_path = os.path.join(feat_dir, f"{split}.pt")
    torch.save(merged, out_path)
    print(f"  Merged {len(shards)} shards → {out_path} "
          f"(N={merged['cls_tokens'].shape[0]})")

    # Cleanup shards
    for i in range(n_shards):
        path = os.path.join(feat_dir, f"{split}_shard{i}.pt")
        if os.path.isfile(path):
            os.remove(path)


# ============================================================================
# Worker entry point (called per GPU subprocess)
# ============================================================================

def worker_main(args):
    """Single GPU worker: extract one shard."""
    extract_shard(args.model_name, args.split, args.shard_id,
                  args.n_shards, args)


# ============================================================================
# Launcher
# ============================================================================

def launch_parallel(args):
    for model_name in args.models:
        for split in ["train", "val"]:
            print(f"\n{'='*68}")
            print(f"Extracting: {model_name} / {split} ({args.n_gpus} GPUs)")
            print('=' * 68)

            procs = []
            log_files = []
            os.makedirs(args.output_dir, exist_ok=True)

            for gpu_id in range(args.n_gpus):
                log_path = os.path.join(
                    args.output_dir,
                    f"log_extract_{model_name}_{split}_gpu{gpu_id}.txt")
                log_f = open(log_path, "w")

                cmd = [
                    sys.executable, "-u", __file__,
                    "--mode", "worker",
                    "--model_name", model_name,
                    "--split", split,
                    "--shard_id", str(gpu_id),
                    "--n_shards", str(args.n_gpus),
                    "--gpu", str(gpu_id),
                    "--imagenet_root", args.imagenet_root,
                    "--image_size", str(args.image_size),
                    "--batch_size", str(args.batch_size),
                    "--num_workers", str(args.num_workers),
                    "--output_dir", args.output_dir,
                ]
                # stdout → 터미널(실시간 진행도), stderr → 로그파일
                p = subprocess.Popen(cmd, stderr=log_f)
                procs.append(p)
                log_files.append(log_f)
                time.sleep(2)

            # Wait
            t0 = time.time()
            for i, p in enumerate(procs):
                p.wait()
                elapsed = (time.time() - t0) / 60
                status = "OK" if p.returncode == 0 else f"FAILED({p.returncode})"
                print(f"  GPU {i}: {status}  ({elapsed:.1f} min)")
                log_files[i].close()

            # Merge
            print(f"  Merging shards...")
            merge_shards(model_name, split, args.n_gpus, args.output_dir)

    print(f"\nDone. Features saved in {args.output_dir}/")
    print(f"Next: python phase1_train_linear.py --feature_dir {args.output_dir}")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="launcher",
                   choices=["launcher", "worker"])
    # Launcher args
    p.add_argument("--models", nargs="+",
                   default=["dinov2_vitl14", "dinov2_vitl14_reg"])
    p.add_argument("--n_gpus", type=int, default=5)
    # Common args
    p.add_argument("--imagenet_root", default="/nas/datahub/imagenet")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--output_dir", default="./features_phase1")
    # Worker-only args
    p.add_argument("--model_name", default="")
    p.add_argument("--split", default="")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    if args.mode == "worker":
        worker_main(args)
    else:
        launch_parallel(args)


if __name__ == "__main__":
    main()
