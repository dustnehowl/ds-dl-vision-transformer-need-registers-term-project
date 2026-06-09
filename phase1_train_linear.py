"""
=============================================================================
 Phase 1 Linear Classifier Training — on pre-extracted features

 미리 추출한 feature 위에서 linear classifier 학습.
 DINOv2 best config 고정: 4 blocks + avgpool (5 × D = 5120 dim)
 GPU 1개, 메모리만 있으면 빠르게 완료.

 Usage:
   python phase1_train_linear.py --gpu 4
=============================================================================
"""

import os
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def build_features(cls_tokens, avg_patches):
    """DINOv2 best config: 4 blocks CLS concat + avgpool.

    cls_tokens: [N, 4, D]
    avg_patches: [N, D]
    → [N, 5D]  (4 CLS + 1 mean patch)
    """
    feat = cls_tokens.reshape(cls_tokens.shape[0], -1)  # [N, 4D]
    feat = torch.cat([feat, avg_patches], dim=-1)        # [N, 5D]
    return feat.float()


def train_and_eval(train_X, train_y, val_X, val_y,
                   lr, num_classes, device,
                   epochs=10, batch_size=1024):
    """Train linear classifier, return val accuracy."""
    in_dim = train_X.shape[1]
    linear = nn.Linear(in_dim, num_classes).to(device)
    nn.init.normal_(linear.weight, mean=0.0, std=0.01)
    nn.init.zeros_(linear.bias)

    optimizer = torch.optim.SGD(linear.parameters(),
                                 lr=lr, momentum=0.9, weight_decay=0)
    n_iter = epochs * (len(train_X) // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iter)

    train_X = train_X.to(device)
    train_y = train_y.to(device)

    linear.train()
    pbar = tqdm(range(epochs), desc="train", dynamic_ncols=True)
    for epoch in pbar:
        perm = torch.randperm(len(train_X), device=device)
        running_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_X) - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(linear(train_X[idx]), train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            n_batches += 1
        pbar.set_postfix({"loss": f"{running_loss/n_batches:.3f}",
                          "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    # Eval
    linear.eval()
    val_X = val_X.to(device)
    val_y = val_y.to(device)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(val_X), batch_size):
            preds = linear(val_X[i:i + batch_size]).argmax(dim=1)
            correct += (preds == val_y[i:i + batch_size]).sum().item()
    return 100.0 * correct / len(val_y)


def run_one_model(model_name, args, device):
    print(f"\n{'='*68}")
    print(f"Model: {model_name}")
    print('=' * 68)
    t0 = time.time()

    feat_dir = os.path.join(args.feature_dir, model_name)
    train_data = torch.load(os.path.join(feat_dir, "train.pt"),
                            map_location="cpu", weights_only=True)
    val_data = torch.load(os.path.join(feat_dir, "val.pt"),
                          map_location="cpu", weights_only=True)

    embed_dim = train_data['cls_tokens'].shape[2]
    print(f"  Train: {train_data['cls_tokens'].shape[0]} images")
    print(f"  Val:   {val_data['cls_tokens'].shape[0]} images")
    print(f"  Embed dim: {embed_dim}")

    num_classes = int(train_data['labels'].max().item()) + 1

    # Best config: 4 blocks + avgpool → 5 × D
    train_X = build_features(train_data['cls_tokens'], train_data['avg_patches'])
    val_X = build_features(val_data['cls_tokens'], val_data['avg_patches'])
    train_y = train_data['labels']
    val_y = val_data['labels']
    print(f"  Config: 4blocks + avgpool → feature dim = {train_X.shape[1]}")

    # LR scaling: lr × batch_size / 256
    lr_scaled = args.lr * args.train_batch_size / 256.0
    print(f"  LR: {args.lr} (scaled: {lr_scaled:.5f})")

    acc = train_and_eval(
        train_X, train_y, val_X, val_y,
        lr_scaled, num_classes, device,
        args.epochs, args.train_batch_size,
    )

    elapsed = (time.time() - t0) / 60
    print(f"\n  -> {model_name} Top-1 = {acc:.2f}%   ({elapsed:.1f} min)")

    return round(acc, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_dir", default="./features_phase1")
    p.add_argument("--models", nargs="+",
                   default=["dinov2_vitl14", "dinov2_vitl14_reg"])
    p.add_argument("--lr", type=float, default=0.01,
                   help="Base learning rate (before batch scaling)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train_batch_size", type=int, default=1024)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", default="./results_phase1_v2.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"Device: {device}")
    print(f"Feature dir: {args.feature_dir}")
    print(f"Config: 4blocks + avgpool (DINOv2 best)")
    print(f"LR: {args.lr}, epochs: {args.epochs}")

    all_results = {}
    for m in args.models:
        try:
            all_results[m] = run_one_model(m, args, device)
        except Exception as e:
            print(f"  ERROR for {m}: {e}")
            import traceback; traceback.print_exc()
            all_results[m] = None
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*64}")
    print("FINAL — ImageNet Linear Classification (Top-1, %)")
    print('=' * 64)
    for m, acc in all_results.items():
        if acc is not None:
            print(f"  {m:32}  {acc:>8.2f}")
        else:
            print(f"  {m:32}  {'FAILED':>8}")

    valid = [v for v in all_results.values() if v is not None]
    if len(valid) == 2:
        diff = valid[1] - valid[0]
        print(f"\n  Delta (reg - no_reg) = {diff:+.2f}%")
        print(f"  Paper Table 2 reports: 84.3 -> 84.8  (Delta = +0.5)")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
