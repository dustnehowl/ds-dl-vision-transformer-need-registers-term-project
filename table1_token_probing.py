"""
=============================================================================
 Vision Transformers Need Registers (Darcet et al., ICLR 2024)
 Table 1 재현 (개선판): Token별 Linear Probing

 목표:  CLS / normal patch / outlier patch token 각각으로 linear probing
        → outlier token이 global information을 carry한다는 trend 확인

 Chen et al. 2020 (SimCLR) 평가 프로토콜을 가능한 한 따름.

 따라가지 못한 점:
   - Birdsnap   : 원본 데이터 호스팅 중단 → CUB200으로 대체
   - VOC2007    : multi-label → 평가 파이프라인이 다름 (제외)
   - Stanford Cars: torchvision auto-download 불가 → 수동 다운로드 시에만 사용
=============================================================================
"""

import os
import argparse
import random
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

# C sweep: paper-style log-spaced regularization values
C_SWEEP = np.logspace(-6, 2, 9).tolist()  # [1e-6, 1e-5, ..., 1e2]

DATASET_INFO = {
    # Core (torchvision native, auto-downloadable)
    "CIFAR10":      {"num_classes": 10,   "auto": True},
    "CIFAR100":     {"num_classes": 100,  "auto": True},
    "Aircraft":     {"num_classes": 100,  "auto": True},
    "DTD":          {"num_classes": 47,   "auto": True},
    "Flowers102":   {"num_classes": 102,  "auto": True},
    "Food101":      {"num_classes": 101,  "auto": True},
    "Pets":         {"num_classes": 37,   "auto": True},
    "SUN397":       {"num_classes": 397,  "auto": True},
    "Caltech101":   {"num_classes": 101,  "auto": True},

    # Substitute / manual
    "CUB200":       {"num_classes": 200,  "auto": False,
                     "note": "Birdsnap 대체. data_root/CUB_200_2011/{train,test}/<class>/..."},
    "StanfordCars": {"num_classes": 196,  "auto": False,
                     "note": "data_root/stanford_cars/{train,test}/<class>/..."},
}


# ============================================================================
# Data
# ============================================================================

def get_transform(image_size=224):
    """DINOv2 표준 전처리. Grayscale 이미지(Caltech101, DTD 등)는 RGB로 변환."""
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(
            int(image_size * 256 / 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _deterministic_class_split(dataset, split, n_train_per_class, seed=0):
    """공식 split이 없는 데이터셋 (SUN397, Caltech101) 용 per-class random split."""
    # target 추출
    if hasattr(dataset, "_labels"):
        targets = list(dataset._labels)            # SUN397
    elif hasattr(dataset, "y"):
        targets = list(dataset.y)                  # Caltech101
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]

    rng = np.random.RandomState(seed)
    by_class = {}
    for idx, y in enumerate(targets):
        by_class.setdefault(int(y), []).append(idx)

    train_idx, test_idx = [], []
    for _, idxs in by_class.items():
        rng.shuffle(idxs)
        train_idx.extend(idxs[:n_train_per_class])
        test_idx.extend(idxs[n_train_per_class:])

    chosen = train_idx if split == "train" else test_idx
    return Subset(dataset, chosen)


def load_dataset(name, root, split, image_size=224):
    """train+val을 합쳐 'train'으로, 공식 test를 'test'로 반환."""
    tf = get_transform(image_size)

    if name in ("CIFAR10", "CIFAR100"):
        cls = datasets.CIFAR10 if name == "CIFAR10" else datasets.CIFAR100
        return cls(root=root, train=(split == "train"), download=True, transform=tf)

    if name == "Aircraft":
        if split == "train":
            tr = datasets.FGVCAircraft(root=root, split="train", download=True, transform=tf)
            va = datasets.FGVCAircraft(root=root, split="val",   download=True, transform=tf)
            return ConcatDataset([tr, va])
        return datasets.FGVCAircraft(root=root, split="test", download=True, transform=tf)

    if name == "DTD":
        # DTD에는 10 partitions, 표준은 partition=1
        return datasets.DTD(root=root,
                            split=("train" if split == "train" else "test"),
                            partition=1, download=True, transform=tf)

    if name == "Flowers102":
        if split == "train":
            tr = datasets.Flowers102(root=root, split="train", download=True, transform=tf)
            va = datasets.Flowers102(root=root, split="val",   download=True, transform=tf)
            return ConcatDataset([tr, va])
        return datasets.Flowers102(root=root, split="test", download=True, transform=tf)

    if name == "Food101":
        return datasets.Food101(root=root,
                                split=("train" if split == "train" else "test"),
                                download=True, transform=tf)

    if name == "Pets":
        return datasets.OxfordIIITPet(root=root,
                                      split=("trainval" if split == "train" else "test"),
                                      download=True, transform=tf)

    if name == "SUN397":
        full = datasets.SUN397(root=root, download=True, transform=tf)
        return _deterministic_class_split(full, split, n_train_per_class=50)

    if name == "Caltech101":
        full = datasets.Caltech101(root=root, download=True, transform=tf)
        return _deterministic_class_split(full, split, n_train_per_class=30)

    if name == "CUB200":
        sub = "train" if split == "train" else "test"
        return datasets.ImageFolder(os.path.join(root, "CUB_200_2011", sub), transform=tf)

    if name == "StanfordCars":
        sub = "train" if split == "train" else "test"
        return datasets.ImageFolder(os.path.join(root, "stanford_cars", sub), transform=tf)

    raise ValueError(f"Unknown dataset: {name}")


# ============================================================================
# Model & feature extraction
# ============================================================================

def load_dinov2(model_name, device):
    print(f"Loading {model_name}...")
    model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params={n_params:.1f}M  embed_dim={model.embed_dim}  "
          f"patch_size={model.patch_size}")
    return model


@torch.no_grad()
def extract_features(model, dataloader, device):
    """
    모든 이미지에 대해 추출:
      - CLS token       : [N, D]
      - patch tokens    : list of [P, D]  (이미지마다 보존, outlier 식별 위해)
      - patch norms     : list of [P]
      - labels          : [N]

    핵심: outlier 식별은 x_prenorm (LayerNorm 이전)에서 수행해야 함.
    """
    all_cls, all_patches, all_norms, all_labels = [], [], [], []
    first = True

    for images, targets in tqdm(dataloader, desc="extract"):
        images = images.to(device, non_blocking=True)
        out = model.forward_features(images)

        if first:
            print(f"    forward_features keys: {list(out.keys())}")
            first = False

        if "x_prenorm" in out:
            tokens = out["x_prenorm"]                     # [B, 1+P, D]
            cls = tokens[:, 0]
            patches = tokens[:, 1:]
        else:
            # fallback (구버전 hub model)
            cls = out["x_norm_clstoken"]
            patches = out["x_norm_patchtokens"]
            print("WARNING: x_prenorm not available; outlier detection may underestimate.")

        norms = patches.norm(dim=-1)                      # [B, P]

        all_cls.append(cls.cpu())
        all_labels.append(targets)
        for i in range(images.size(0)):
            all_patches.append(patches[i].cpu())
            all_norms.append(norms[i].cpu())

    return (torch.cat(all_cls, dim=0),
            all_patches, all_norms,
            torch.cat(all_labels, dim=0))


def select_token(patches_list, norms_list, threshold, kind, seed):
    """각 이미지에서 normal/outlier patch 1개 무작위 선택."""
    rng = np.random.RandomState(seed)
    N = len(patches_list)
    D = patches_list[0].size(-1)
    out = torch.zeros(N, D)
    valid = torch.zeros(N, dtype=torch.bool)

    for i in range(N):
        if kind == "outlier":
            mask = norms_list[i] > threshold
        else:
            mask = norms_list[i] <= threshold
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            j = idx[rng.randint(len(idx))]
            out[i] = patches_list[i][j]
            valid[i] = True
    return out, valid


# ============================================================================
# Linear probing with C sweep
# ============================================================================

def linear_probe_sweep(train_X, train_y, test_X, test_y,
                       device="cuda", val_frac=0.2, seed=0, max_iter=200):
    """
    GPU-accelerated linear probing with C sweep (Kornblith/Chen style):
      1) train → (sub-train, val) split
      2) C_SWEEP의 모든 값으로 sub-train 학습 → val에서 best C 선택
      3) best_C로 train 전체에서 retrain → test 평가

    Backend: PyTorch LBFGS on GPU.
    Objective는 sklearn과 동일:  sum_i CE(x_i, y_i; w) + (1/2C) ||w||^2
    """
    import torch.nn as nn
    import torch.nn.functional as F

    # numpy/torch 모두 받기
    def to_dev(x, dtype):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(device=device, dtype=dtype)

    X_tr_all = to_dev(train_X, torch.float32)
    y_tr_all = to_dev(train_y, torch.long)
    X_te     = to_dev(test_X,  torch.float32)
    y_te     = to_dev(test_y,  torch.long)

    n_classes = int(y_tr_all.max().item()) + 1
    D = X_tr_all.size(1)

    # train/val split
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(X_tr_all.size(0), generator=g).to(device)
    n_val = max(1, int(X_tr_all.size(0) * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, y_tr = X_tr_all[tr_idx], y_tr_all[tr_idx]
    X_val, y_val = X_tr_all[val_idx], y_tr_all[val_idx]

    def fit(X, y, C, n_iter):
        """L2-regularized logistic regression via PyTorch LBFGS on GPU."""
        model = nn.Linear(D, n_classes, bias=True).to(device)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        opt = torch.optim.LBFGS(
            model.parameters(),
            max_iter=n_iter,
            history_size=10,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-5,
            tolerance_change=1e-9,
        )

        def closure():
            opt.zero_grad()
            logits = model(X)
            # sklearn-equivalent objective: sum CE + (1/(2C)) ||w||^2
            ce = F.cross_entropy(logits, y, reduction="sum")
            l2 = 0.5 * (model.weight ** 2).sum() / C
            loss = ce + l2
            loss.backward()
            return loss

        opt.step(closure)
        return model

    @torch.no_grad()
    def acc(model, X, y):
        return (model(X).argmax(dim=1) == y).float().mean().item()

    # sweep C
    best_C, best_val = None, -1.0
    for C in C_SWEEP:
        m = fit(X_tr, y_tr, C, n_iter=max_iter)
        v = acc(m, X_val, y_val)
        if v > best_val:
            best_val, best_C = v, C

    # retrain on full train with best C, more iterations
    final = fit(X_tr_all, y_tr_all, best_C, n_iter=max_iter * 2)
    test_acc = acc(final, X_te, y_te) * 100.0
    return test_acc, best_C


# ============================================================================
# Threshold helpers
# ============================================================================

def find_threshold(norms_list, target_ratio=0.0237):
    """상위 target_ratio (paper: 2.37%) 분위수 = threshold."""
    all_norms = torch.cat(norms_list).float().numpy()
    return float(np.quantile(all_norms, 1.0 - target_ratio))


def report_norms(norms_list, threshold):
    all_norms = torch.cat(norms_list).float()
    out_ratio = (all_norms > threshold).float().mean().item() * 100
    n_with = sum(1 for n in norms_list if (n > threshold).any())
    # numpy.quantile은 큰 텐서 제약 없음 (torch.quantile은 2^24 limit)
    p99 = float(np.quantile(all_norms.numpy(), 0.99))
    print(f"  norm: mean={all_norms.mean():.1f} max={all_norms.max():.1f} "
          f"p99={p99:.1f}")
    print(f"  threshold={threshold:.1f} → {out_ratio:.2f}% of patches are outliers, "
          f"present in {n_with}/{len(norms_list)} images "
          f"({100*n_with/len(norms_list):.1f}%)")


# ============================================================================
# Main
# ============================================================================

def run(args):
    device = f"cuda:{args.gpu}"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = load_dinov2(args.model, device)
    n_patches = (args.image_size // model.patch_size) ** 2
    print(f"  image_size={args.image_size} → {n_patches} patches per image")
    print(f"  C sweep over: {[f'{c:.0e}' for c in C_SWEEP]}")

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_{args.results_suffix}" if args.results_suffix else ""
    out_path = os.path.join(
        args.output_dir,
        f"results_{args.model}_r{args.image_size}{suffix}.json"
    )
    all_results = {}

    for name in args.datasets:
        print(f"\n{'='*68}\nDataset: {name}\n{'='*68}")
        try:
            train_ds = load_dataset(name, args.data_root, "train", args.image_size)
            test_ds  = load_dataset(name, args.data_root, "test",  args.image_size)
        except Exception as e:
            print(f"  [SKIP] {type(e).__name__}: {e}")
            if not DATASET_INFO.get(name, {}).get("auto", False):
                print(f"  Note: {DATASET_INFO[name].get('note', '')}")
            continue

        print(f"  train={len(train_ds)}  test={len(test_ds)}")
        loader_kw = dict(batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
        train_loader = DataLoader(train_ds, **loader_kw)
        test_loader  = DataLoader(test_ds,  **loader_kw)

        # 1) Feature extraction
        print("  [train] extracting features...")
        tr_cls, tr_patches, tr_norms, tr_y = extract_features(model, train_loader, device)
        print("  [test] extracting features...")
        te_cls, te_patches, te_norms, te_y = extract_features(model, test_loader,  device)
        tr_y_np, te_y_np = tr_y.numpy(), te_y.numpy()

        # 2) Threshold
        thr = (find_threshold(tr_norms, args.outlier_ratio)
               if args.auto_threshold else args.norm_threshold)
        report_norms(tr_norms, thr)

        # 3) Linear probe: CLS
        print("  [CLS] C sweep...")
        cls_acc, cls_C = linear_probe_sweep(
            tr_cls.numpy(), tr_y_np, te_cls.numpy(), te_y_np,
            device=device, seed=args.seed)
        print(f"    → CLS = {cls_acc:.2f}%  (best C={cls_C:.0e})")

        # 4) Linear probe: normal / outlier (multiple trials)
        normal_accs, outlier_accs = [], []
        for trial in range(args.num_trials):
            s = args.seed + trial

            # normal patch
            trX, trV = select_token(tr_patches, tr_norms, thr, "normal", s)
            teX, teV = select_token(te_patches, te_norms, thr, "normal", s)
            m_tr, m_te = trV.numpy(), teV.numpy()
            if m_tr.sum() >= 100 and m_te.sum() >= 100:
                acc, _ = linear_probe_sweep(
                    trX[m_tr].numpy(), tr_y_np[m_tr],
                    teX[m_te].numpy(), te_y_np[m_te],
                    device=device, seed=s)
                normal_accs.append(acc)
                print(f"    trial {trial+1}/{args.num_trials}  normal  = {acc:.2f}%  "
                      f"(samples: tr={m_tr.sum()}/{len(m_tr)}, te={m_te.sum()}/{len(m_te)})")

            # outlier patch
            trX, trV = select_token(tr_patches, tr_norms, thr, "outlier", s)
            teX, teV = select_token(te_patches, te_norms, thr, "outlier", s)
            m_tr, m_te = trV.numpy(), teV.numpy()
            if m_tr.sum() >= 100 and m_te.sum() >= 100:
                acc, _ = linear_probe_sweep(
                    trX[m_tr].numpy(), tr_y_np[m_tr],
                    teX[m_te].numpy(), te_y_np[m_te],
                    device=device, seed=s)
                outlier_accs.append(acc)
                print(f"    trial {trial+1}/{args.num_trials}  outlier = {acc:.2f}%  "
                      f"(samples: tr={m_tr.sum()}/{len(m_tr)}, te={m_te.sum()}/{len(m_te)})")
            else:
                print(f"    trial {trial+1}/{args.num_trials}  outlier: "
                      f"insufficient samples (tr={m_tr.sum()}, te={m_te.sum()}). "
                      f"Try higher resolution or larger model.")

        all_results[name] = {
            "cls":           round(float(cls_acc), 2),
            "normal_mean":   round(float(np.mean(normal_accs)),  2) if normal_accs  else None,
            "normal_std":    round(float(np.std(normal_accs)),   2) if len(normal_accs)  > 1 else 0.0,
            "outlier_mean":  round(float(np.mean(outlier_accs)), 2) if outlier_accs else None,
            "outlier_std":   round(float(np.std(outlier_accs)),  2) if len(outlier_accs) > 1 else 0.0,
            "threshold":     round(float(thr), 2),
            "n_train":       len(train_ds),
            "n_test":        len(test_ds),
        }

        # Incremental save
        with open(out_path, "w") as f:
            json.dump({
                "model": args.model,
                "image_size": args.image_size,
                "auto_threshold": args.auto_threshold,
                "outlier_ratio_target": args.outlier_ratio,
                "results": all_results,
            }, f, indent=2)

    # ------------------------------------------------------------------
    # Final table
    # ------------------------------------------------------------------
    print(f"\n{'='*76}")
    print(f"FINAL — {args.model}, image_size={args.image_size}")
    print('='*76)
    header = f"{'dataset':<14} {'CLS':>8} {'normal':>14} {'outlier':>14}   Δ(out-norm)"
    print(header)
    print('-' * len(header))
    for name, r in all_results.items():
        cls = r["cls"]
        if r["normal_mean"] is not None and r["outlier_mean"] is not None:
            nm = f"{r['normal_mean']:.1f}±{r['normal_std']:.1f}"
            om = f"{r['outlier_mean']:.1f}±{r['outlier_std']:.1f}"
            d  = r["outlier_mean"] - r["normal_mean"]
            print(f"{name:<14} {cls:>8.1f} {nm:>14} {om:>14}   {d:+.1f}")
        else:
            nm = f"{r['normal_mean']:.1f}" if r['normal_mean']  is not None else "N/A"
            om = f"{r['outlier_mean']:.1f}" if r['outlier_mean'] is not None else "N/A"
            print(f"{name:<14} {cls:>8.1f} {nm:>14} {om:>14}   {'—':>10}")
    print(f"\nSaved → {out_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dinov2_vitl14",
                   choices=["dinov2_vits14", "dinov2_vitb14",
                            "dinov2_vitl14", "dinov2_vitg14"])
    p.add_argument("--datasets", nargs="+",
                   default=["CIFAR10", "CIFAR100", "Aircraft", "DTD",
                            "Flowers102", "Food101", "Pets",
                            "SUN397", "Caltech101", "CUB200"])
    p.add_argument("--data_root", default="./data")
    p.add_argument("--image_size", type=int, default=224,
                   help="224 (fast) | 448 (more outliers)")
    p.add_argument("--norm_threshold", type=float, default=150.0,
                   help="Used only when --auto_threshold is OFF")
    p.add_argument("--auto_threshold", action="store_true",
                   help="Derive threshold from train norm distribution (RECOMMENDED)")
    p.add_argument("--outlier_ratio", type=float, default=0.0237,
                   help="Target outlier ratio for --auto_threshold (paper: 2.37%%)")
    p.add_argument("--num_trials", type=int, default=3,
                   help="Random token selection trials for normal/outlier")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--results_suffix", default="",
                   help="Suffix added to output JSON filename (for parallel runs)")
    args = p.parse_args()

    run(args)