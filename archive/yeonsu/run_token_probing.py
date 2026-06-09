"""
=============================================================================
 Vision Transformers Need Registers (Darcet et al., ICLR 2024)
 Experiment 1 재현: Table 1 - Token별 Linear Probing
 
 CLS token / normal patch token / outlier patch token 각각으로
 image classification linear probing 성능 비교
=============================================================================
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
from pathlib import Path


# ============================================================================
# 설정
# ============================================================================

DATASETS_CONFIG = {
    "CIFAR10": {
        "class": "CIFAR10",
        "num_classes": 10,
        "builtin": True,
    },
    "CIFAR100": {
        "class": "CIFAR100",
        "num_classes": 100,
        "builtin": True,
    },
    "Aircraft": {
        "class": "FGVCAircraft",
        "num_classes": 100,
        "builtin": True,
    },
    "DTD": {
        "class": "DTD",
        "num_classes": 47,
        "builtin": True,
    },
    "Flowers102": {
        "class": "Flowers102",
        "num_classes": 102,
        "builtin": True,
    },
    "Food101": {
        "class": "Food101",
        "num_classes": 101,
        "builtin": True,
    },
    "Pets": {
        "class": "OxfordIIITPet",
        "num_classes": 37,
        "builtin": True,
    },
    "CUB200": {
        "class": "ImageFolder",
        "num_classes": 200,
        "builtin": False,
        "note": "CUB-200-2011 - ImageFolder 형식으로 준비 필요",
    },
    "ImageNet1k": {
        "class": "ImageFolder",
        "num_classes": 1000,
        "builtin": False,
        "note": "ImageNet-1k - ImageFolder 형식 (/train, /val)",
    },
}


# ============================================================================
# Feature Extraction
# ============================================================================

def get_transform(image_size=224):
    """DINOv2 표준 전처리"""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_dataset(name, root, split="test", image_size=224):
    """데이터셋 로드"""
    transform = get_transform(image_size)
    
    if name == "CIFAR10":
        is_train = (split == "train")
        ds = datasets.CIFAR10(root=root, train=is_train, download=True, transform=transform)
    elif name == "CIFAR100":
        is_train = (split == "train")
        ds = datasets.CIFAR100(root=root, train=is_train, download=True, transform=transform)
    elif name == "Aircraft":
        ds = datasets.FGVCAircraft(root=root, split=split if split != "test" else "test",
                                    download=True, transform=transform)
    elif name == "DTD":
        split_map = {"train": "train", "test": "test", "val": "val"}
        ds = datasets.DTD(root=root, split=split_map.get(split, "test"),
                          download=True, transform=transform)
    elif name == "Flowers102":
        ds = datasets.Flowers102(root=root, split=split if split != "val" else "test",
                                  download=True, transform=transform)
    elif name == "Food101":
        ds = datasets.Food101(root=root, split="train" if split == "train" else "test",
                               download=True, transform=transform)
    elif name == "Pets":
        ds = datasets.OxfordIIITPet(root=root, split="trainval" if split == "train" else "test",
                                     download=True, transform=transform)
    elif name == "ImageNet1k":
        split_dir = os.path.join(root, "imagenet", "train" if split == "train" else "val")
        ds = datasets.ImageFolder(split_dir, transform=transform)
    elif name == "CUB200":
        split_dir = os.path.join(root, "CUB_200_2011", "train" if split == "train" else "test")
        ds = datasets.ImageFolder(split_dir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return ds


def load_dinov2_model(model_name="dinov2_vitl14", device="cuda"):
    """
    DINOv2 모델 로드
    
    사용 가능 모델:
    - dinov2_vits14: ViT-S/14 (작음, 빠름, outlier 없을 수 있음)
    - dinov2_vitb14: ViT-B/14
    - dinov2_vitl14: ViT-L/14 (권장 - outlier 있고 4090에 적합)
    - dinov2_vitg14: ViT-g/14 (가장 큼 - 논문 Table 1 기준 모델)
    """
    print(f"Loading model: {model_name}")
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model = model.to(device)
    model.eval()
    
    # 모델 정보 출력
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {num_params:.1f}M")
    print(f"  Embed dim: {model.embed_dim}")
    print(f"  Num heads: {model.num_heads}")
    print(f"  Patch size: {model.patch_size}")
    
    return model


@torch.no_grad()
def extract_features(model, dataloader, device="cuda", norm_threshold=150.0):
    """
    모든 이미지에 대해 CLS token, patch tokens, norms 추출
    
    중요: DINOv2의 forward_features()는 두 가지 출력을 제공:
      - x_norm_*: 최종 LayerNorm 이후 (norm이 좁은 범위로 압축됨)
      - x_prenorm: LayerNorm 이전 raw output (논문의 outlier norm 측정 기준)
    
    논문의 Figure 3에서 norm > 150인 outlier는 prenorm 기준입니다.
    Linear probing에는 prenorm features를 사용합니다.
    
    Returns:
        cls_features: [N, D]
        patch_features: list of [num_patches, D] per image
        patch_norms: list of [num_patches] per image  (prenorm 기준)
        labels: [N]
    """
    all_cls = []
    all_patch_features = []
    all_patch_norms = []
    all_labels = []
    
    first_batch = True
    for images, targets in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        
        # DINOv2 forward
        output = model.forward_features(images)
        
        # 첫 배치에서 출력 키 확인
        if first_batch:
            print(f"    forward_features output keys: {list(output.keys())}")
            first_batch = False
        
        # ====== 핵심 수정: prenorm 사용 ======
        # x_prenorm: LayerNorm 이전의 전체 시퀀스 [B, 1+N_patches, D]
        # x_prenorm[:, 0] = CLS token (prenorm)
        # x_prenorm[:, 1:] = patch tokens (prenorm)
        
        if 'x_prenorm' in output:
            prenorm = output['x_prenorm']
            cls_tokens = prenorm[:, 0]       # [B, D] - CLS (prenorm)
            patch_tokens = prenorm[:, 1:]    # [B, N_patches, D] - patches (prenorm)
            
            # Norm 계산: prenorm에서 해야 outlier가 보임
            norms = patch_tokens.norm(dim=-1)  # [B, N_patches]
            
            if len(all_cls) == 0:  # 첫 배치
                print(f"    x_prenorm shape: {prenorm.shape}")
                print(f"    Sample prenorm norms - min: {norms.min():.1f}, "
                      f"max: {norms.max():.1f}, mean: {norms.mean():.1f}")
        else:
            # fallback: x_prenorm이 없는 경우 (구버전 등)
            print("WARNING: x_prenorm not found, falling back to x_norm_patchtokens")
            print("         Outlier detection may not work properly!")
            cls_tokens = output['x_norm_clstoken']
            patch_tokens = output['x_norm_patchtokens']
            norms = patch_tokens.norm(dim=-1)
        
        all_cls.append(cls_tokens.cpu())
        all_labels.append(targets)
        
        # 이미지별로 저장 (outlier/normal 구분 위해)
        for i in range(images.shape[0]):
            all_patch_features.append(patch_tokens[i].cpu())
            all_patch_norms.append(norms[i].cpu())
    
    cls_features = torch.cat(all_cls, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return cls_features, all_patch_features, all_patch_norms, labels


def select_tokens(patch_features, patch_norms, norm_threshold, token_type="normal", seed=42):
    """
    각 이미지에서 token_type에 해당하는 patch token 1개를 random 선택
    
    Args:
        token_type: "normal" | "outlier"
    Returns:
        selected_features: [N, D] or None (해당 타입 토큰이 없는 이미지 존재 시)
        valid_mask: [N] bool - 해당 타입 토큰이 있는 이미지
    """
    rng = np.random.RandomState(seed)
    N = len(patch_features)
    D = patch_features[0].shape[1]
    
    selected = torch.zeros(N, D)
    valid = torch.zeros(N, dtype=torch.bool)
    
    for i in range(N):
        norms = patch_norms[i]
        feats = patch_features[i]
        
        if token_type == "outlier":
            mask = norms > norm_threshold
        else:  # normal
            mask = norms <= norm_threshold
        
        indices = mask.nonzero(as_tuple=True)[0]
        
        if len(indices) > 0:
            idx = indices[rng.randint(len(indices))]
            selected[i] = feats[idx]
            valid[i] = True
    
    return selected, valid


# ============================================================================
# Linear Probing
# ============================================================================

def linear_probe(train_features, train_labels, test_features, test_labels, 
                 C=0.01, max_iter=1000):
    """
    sklearn LogisticRegression으로 linear probing
    논문에서는 logistic regression classifier 사용
    """
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1,
        verbose=0,
    )
    clf.fit(train_features_scaled, train_labels)
    
    acc = clf.score(test_features_scaled, test_labels) * 100
    return acc


# ============================================================================
# Norm 분석
# ============================================================================

def analyze_norms(patch_norms, norm_threshold=150.0):
    """Norm 분포 분석 - threshold 결정에 도움"""
    all_norms = torch.cat(patch_norms)
    
    print("\n" + "=" * 60)
    print("Patch Token Norm 분석")
    print("=" * 60)
    print(f"  Total tokens: {len(all_norms)}")
    print(f"  Mean norm:    {all_norms.mean():.2f}")
    print(f"  Std norm:     {all_norms.std():.2f}")
    print(f"  Min norm:     {all_norms.min():.2f}")
    print(f"  Max norm:     {all_norms.max():.2f}")
    print(f"  Median norm:  {all_norms.median():.2f}")
    
    # Percentile 분석
    for p in [90, 95, 99, 99.5, 99.9]:
        val = torch.quantile(all_norms.float(), p / 100).item()
        print(f"  {p}th percentile: {val:.2f}")
    
    # 현재 threshold 기준 outlier 비율
    outlier_ratio = (all_norms > norm_threshold).float().mean().item() * 100
    print(f"\n  Threshold={norm_threshold}: outlier 비율 = {outlier_ratio:.2f}%")
    print(f"  (논문 기준: ~2.37% for DINOv2-g)")
    
    # Outlier가 있는 이미지 비율
    images_with_outliers = sum(1 for norms in patch_norms if (norms > norm_threshold).any())
    print(f"  Outlier가 있는 이미지: {images_with_outliers}/{len(patch_norms)} "
          f"({images_with_outliers/len(patch_norms)*100:.1f}%)")
    
    return all_norms


def find_optimal_threshold(patch_norms, target_ratio=0.0237):
    """
    논문의 2.37% 비율에 맞는 threshold를 자동으로 찾기
    (모델마다 절대 norm 값이 다를 수 있으므로)
    """
    all_norms = torch.cat(patch_norms).float()
    threshold = torch.quantile(all_norms, 1.0 - target_ratio).item()
    actual_ratio = (all_norms > threshold).float().mean().item()
    print(f"\n  자동 threshold (상위 {target_ratio*100:.2f}%): {threshold:.2f}")
    print(f"  실제 outlier 비율: {actual_ratio*100:.2f}%")
    return threshold


# ============================================================================
# Main
# ============================================================================

def run_experiment(args):
    device = f"cuda:{args.gpu}"
    
    # 결과 저장
    results = {}
    
    # 1. 모델 로드
    model = load_dinov2_model(args.model, device=device)
    
    # patch 크기에 따른 이미지 크기 결정
    if "14" in args.model:
        image_size = 518  # 518 / 14 = 37 -> 37x37 = 1369 patches (논문 high-res)
        # 또는 224 / 14 = 16 -> 16x16 = 256 patches (표준)
        image_size = 224  # 기본은 224 사용
    else:
        image_size = 224
    
    print(f"\nImage size: {image_size}")
    print(f"Expected patches: {(image_size // 14) ** 2}" if "14" in args.model 
          else f"Expected patches: {(image_size // 16) ** 2}")
    
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # 2. 데이터셋 로드
            train_ds = load_dataset(dataset_name, args.data_root, split="train", 
                                     image_size=image_size)
            test_ds = load_dataset(dataset_name, args.data_root, split="test",
                                    image_size=image_size)
            
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                       shuffle=False, num_workers=args.num_workers,
                                       pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers,
                                      pin_memory=True)
            
            print(f"  Train samples: {len(train_ds)}")
            print(f"  Test samples:  {len(test_ds)}")
            
            # 3. Feature 추출
            print("\n  [Train] Feature extraction...")
            train_cls, train_patches, train_norms, train_labels = \
                extract_features(model, train_loader, device=device)
            
            print("  [Test] Feature extraction...")
            test_cls, test_patches, test_norms, test_labels = \
                extract_features(model, test_loader, device=device)
            
            # 4. Norm 분석 (train set 기준)
            all_norms_analysis = analyze_norms(train_norms, args.norm_threshold)
            
            # threshold 자동 결정 옵션
            if args.auto_threshold:
                threshold = find_optimal_threshold(train_norms)
            else:
                threshold = args.norm_threshold
            
            # 5. Linear Probing - 여러 번 반복 (random token 선택 variance 측정)
            dataset_results = {"CLS": [], "normal": [], "outlier": []}
            
            for trial in range(args.num_trials):
                seed = args.seed + trial
                print(f"\n  --- Trial {trial+1}/{args.num_trials} (seed={seed}) ---")
                
                # [CLS] token
                if trial == 0:  # CLS는 deterministic
                    cls_acc = linear_probe(
                        train_cls.numpy(), train_labels.numpy(),
                        test_cls.numpy(), test_labels.numpy(),
                        C=args.C
                    )
                    print(f"    [CLS] Top-1 Acc: {cls_acc:.1f}%")
                dataset_results["CLS"].append(cls_acc)
                
                # Normal tokens
                train_normal, train_valid = select_tokens(
                    train_patches, train_norms, threshold, "normal", seed=seed)
                test_normal, test_valid = select_tokens(
                    test_patches, test_norms, threshold, "normal", seed=seed)
                
                if train_valid.all() and test_valid.all():
                    normal_acc = linear_probe(
                        train_normal.numpy(), train_labels.numpy(),
                        test_normal.numpy(), test_labels.numpy(),
                        C=args.C
                    )
                    print(f"    [Normal] Top-1 Acc: {normal_acc:.1f}%")
                    dataset_results["normal"].append(normal_acc)
                else:
                    print(f"    [Normal] 일부 이미지에 normal token 없음 "
                          f"(train: {train_valid.sum()}/{len(train_valid)}, "
                          f"test: {test_valid.sum()}/{len(test_valid)})")
                    # valid한 것만으로 진행
                    train_mask = train_valid.numpy()
                    test_mask = test_valid.numpy()
                    if train_mask.sum() > 100 and test_mask.sum() > 100:
                        normal_acc = linear_probe(
                            train_normal[train_mask].numpy(), 
                            train_labels[train_mask].numpy(),
                            test_normal[test_mask].numpy(), 
                            test_labels[test_mask].numpy(),
                            C=args.C
                        )
                        print(f"    [Normal] Top-1 Acc (valid only): {normal_acc:.1f}%")
                        dataset_results["normal"].append(normal_acc)
                
                # Outlier tokens
                train_outlier, train_valid_o = select_tokens(
                    train_patches, train_norms, threshold, "outlier", seed=seed)
                test_outlier, test_valid_o = select_tokens(
                    test_patches, test_norms, threshold, "outlier", seed=seed)
                
                train_mask_o = train_valid_o.numpy()
                test_mask_o = test_valid_o.numpy()
                
                print(f"    [Outlier] 이미지 중 outlier 존재: "
                      f"train={train_mask_o.sum()}/{len(train_mask_o)}, "
                      f"test={test_mask_o.sum()}/{len(test_mask_o)}")
                
                if train_mask_o.sum() > 100 and test_mask_o.sum() > 100:
                    outlier_acc = linear_probe(
                        train_outlier[train_mask_o].numpy(),
                        train_labels[train_mask_o].numpy(),
                        test_outlier[test_mask_o].numpy(),
                        test_labels[test_mask_o].numpy(),
                        C=args.C
                    )
                    print(f"    [Outlier] Top-1 Acc: {outlier_acc:.1f}%")
                    dataset_results["outlier"].append(outlier_acc)
                else:
                    print(f"    [Outlier] Outlier 토큰이 충분하지 않습니다.")
                    print(f"    -> 모델이 충분히 크거나 threshold를 낮춰보세요.")
            
            # 결과 정리
            results[dataset_name] = {}
            for token_type in ["CLS", "normal", "outlier"]:
                accs = dataset_results[token_type]
                if accs:
                    mean_acc = np.mean(accs)
                    std_acc = np.std(accs) if len(accs) > 1 else 0.0
                    results[dataset_name][token_type] = {
                        "mean": round(mean_acc, 1),
                        "std": round(std_acc, 1),
                        "trials": [round(a, 1) for a in accs],
                    }
                    
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 6. 최종 결과 출력
    print("\n\n" + "=" * 80)
    print("최종 결과 (Table 1 재현)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Norm threshold: {threshold}")
    print(f"Trials: {args.num_trials}")
    print()
    
    # 테이블 형식 출력
    header = f"{'Token':>10}"
    for ds_name in results:
        header += f" {ds_name:>10}"
    print(header)
    print("-" * len(header))
    
    for token_type in ["CLS", "normal", "outlier"]:
        row = f"{token_type:>10}"
        for ds_name in results:
            if token_type in results[ds_name]:
                r = results[ds_name][token_type]
                if r["std"] > 0:
                    row += f" {r['mean']:>6.1f}±{r['std']:<3.1f}"
                else:
                    row += f" {r['mean']:>10.1f}"
            else:
                row += f" {'N/A':>10}"
        print(row)
    
    # 결과 JSON 저장
    save_path = os.path.join(args.output_dir, f"results_{args.model}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({
            "model": args.model,
            "norm_threshold": threshold,
            "num_trials": args.num_trials,
            "results": results,
        }, f, indent=2)
    print(f"\n결과 저장: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vision Transformers Need Registers - Table 1 재현"
    )
    
    # 모델
    parser.add_argument("--model", type=str, default="dinov2_vitl14",
                        choices=["dinov2_vits14", "dinov2_vitb14", 
                                 "dinov2_vitl14", "dinov2_vitg14"],
                        help="DINOv2 모델 (기본: vitl14)")
    
    # 데이터
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["CIFAR10", "CIFAR100", "Aircraft", "DTD"],
                        help="평가할 데이터셋 목록")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="데이터 저장 경로")
    
    # Outlier 설정
    parser.add_argument("--norm_threshold", type=float, default=150.0,
                        help="Outlier 판별 norm threshold (기본: 150)")
    parser.add_argument("--auto_threshold", action="store_true",
                        help="상위 2.37%%에 해당하는 threshold 자동 결정")
    
    # Linear probing
    parser.add_argument("--C", type=float, default=0.01,
                        help="LogisticRegression regularization (기본: 0.01)")
    parser.add_argument("--num_trials", type=int, default=5,
                        help="Random token 선택 반복 횟수 (기본: 5)")
    
    # 기타
    parser.add_argument("--batch_size", type=int, default=128,
                        help="배치 사이즈 (기본: 128)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers (기본: 8)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="사용할 GPU 번호 (기본: 0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="결과 저장 경로")
    
    args = parser.parse_args()
    
    # Seed 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    run_experiment(args)