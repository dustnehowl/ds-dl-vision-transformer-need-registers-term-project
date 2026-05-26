"""
=============================================================================
 ImageNet용 Token Linear Probing (메모리 효율 버전)
 
 ImageNet은 1.28M 이미지라 모든 patch token을 RAM에 저장할 수 없음.
 추출과 동시에 normal/outlier 토큰을 선택하여 이미지당 1개씩만 저장.
=============================================================================
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import json


def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_dinov2_model(model_name, device):
    print(f"Loading model: {model_name}")
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model = model.to(device).eval()
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {num_params:.1f}M, Embed dim: {model.embed_dim}")
    return model


@torch.no_grad()
def extract_and_select(model, dataloader, device, norm_threshold, seed=42):
    """
    메모리 효율적 추출: 배치마다 즉시 토큰 선택
    
    각 이미지에서:
      - CLS token (prenorm) 저장
      - Normal patch 중 random 1개 저장
      - Outlier patch 중 random 1개 저장 (없으면 None)
    """
    rng = np.random.RandomState(seed)
    
    all_cls = []
    all_normal = []
    all_outlier = []
    all_labels = []
    has_outlier = []  # 이미지에 outlier가 있는지
    
    # 통계용
    total_tokens = 0
    outlier_tokens = 0
    norm_sum = 0.0
    norm_max = 0.0
    
    first_batch = True
    for images, targets in tqdm(dataloader, desc="Extracting"):
        images = images.to(device)
        output = model.forward_features(images)
        
        prenorm = output['x_prenorm']
        prenorm_patches = prenorm[:, 1:]   # [B, N, D] - for norm calculation
        norms = prenorm_patches.norm(dim=-1)  # [B, N]
        
        # Feature는 post-norm에서 가져옴 (linear probing에 적합)
        cls_tokens = output['x_norm_clstoken']     # [B, D]
        patch_tokens = output['x_norm_patchtokens'] # [B, N, D]
        
        if first_batch:
            print(f"  prenorm shape: {prenorm.shape}")
            print(f"  Norms - min: {norms.min():.1f}, max: {norms.max():.1f}, "
                  f"mean: {norms.mean():.1f}")
            first_batch = False
        
        B = images.shape[0]
        
        # 통계 누적
        total_tokens += norms.numel()
        outlier_tokens += (norms > norm_threshold).sum().item()
        norm_sum += norms.sum().item()
        norm_max = max(norm_max, norms.max().item())
        
        for i in range(B):
            img_norms = norms[i]  # [N]
            img_patches = patch_tokens[i]  # [N, D]
            
            normal_mask = img_norms <= norm_threshold
            outlier_mask = img_norms > norm_threshold
            
            # CLS
            all_cls.append(cls_tokens[i].cpu())
            all_labels.append(targets[i].item())
            
            # Normal token: random 1개
            normal_indices = normal_mask.nonzero(as_tuple=True)[0]
            if len(normal_indices) > 0:
                idx = normal_indices[rng.randint(len(normal_indices))]
                all_normal.append(img_patches[idx].cpu())
            else:
                all_normal.append(torch.zeros(img_patches.shape[1]))
            
            # Outlier token: random 1개
            outlier_indices = outlier_mask.nonzero(as_tuple=True)[0]
            if len(outlier_indices) > 0:
                idx = outlier_indices[rng.randint(len(outlier_indices))]
                all_outlier.append(img_patches[idx].cpu())
                has_outlier.append(True)
            else:
                all_outlier.append(torch.zeros(img_patches.shape[1]))
                has_outlier.append(False)
    
    # 통계 출력
    print(f"\n  Norm 통계:")
    print(f"    Total tokens: {total_tokens:,}")
    print(f"    Mean norm: {norm_sum / total_tokens:.2f}")
    print(f"    Max norm: {norm_max:.2f}")
    print(f"    Outlier 비율 (>{norm_threshold}): "
          f"{outlier_tokens/total_tokens*100:.2f}%")
    print(f"    Outlier 있는 이미지: {sum(has_outlier)}/{len(has_outlier)} "
          f"({sum(has_outlier)/len(has_outlier)*100:.1f}%)")
    
    cls_features = torch.stack(all_cls)
    normal_features = torch.stack(all_normal)
    outlier_features = torch.stack(all_outlier)
    labels = np.array(all_labels)
    has_outlier = np.array(has_outlier)
    
    return cls_features, normal_features, outlier_features, labels, has_outlier


def linear_probe(train_feat, train_labels, test_feat, test_labels, 
                 device="cuda:0", lr=0.1, epochs=100, batch_size=1024, 
                 weight_decay=1e-4):
    """
    PyTorch GPU 기반 Linear Probing
    sklearn LBFGS 대비 10~50배 빠름
    """
    # numpy/tensor 통합 처리
    if isinstance(train_feat, np.ndarray):
        train_feat = torch.from_numpy(train_feat)
    if isinstance(train_labels, np.ndarray):
        train_labels = torch.from_numpy(train_labels)
    if isinstance(test_feat, np.ndarray):
        test_feat = torch.from_numpy(test_feat)
    if isinstance(test_labels, np.ndarray):
        test_labels = torch.from_numpy(test_labels)
    
    train_feat = train_feat.float()
    train_labels = train_labels.long()
    test_feat = test_feat.float()
    test_labels = test_labels.long()
    
    # Feature 정규화 (StandardScaler와 동일)
    mean = train_feat.mean(dim=0)
    std = train_feat.std(dim=0).clamp(min=1e-6)
    train_feat = (train_feat - mean) / std
    test_feat = (test_feat - mean) / std
    
    feat_dim = train_feat.shape[1]
    num_classes = train_labels.max().item() + 1
    
    # Linear classifier
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    nn.init.zeros_(classifier.bias)
    nn.init.normal_(classifier.weight, std=0.01)
    
    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # DataLoader
    train_dataset = TensorDataset(train_feat, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    
    # 학습
    classifier.train()
    for epoch in range(epochs):
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = classifier(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    # 평가
    classifier.eval()
    with torch.no_grad():
        # 큰 test set도 배치로 처리
        correct = 0
        total = 0
        test_dataset = TensorDataset(test_feat, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
        for feats, labels in test_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = classifier(feats)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    acc = correct / total * 100
    return acc


def main():
    parser = argparse.ArgumentParser(description="ImageNet Token Linear Probing")
    parser.add_argument("--model", type=str, default="dinov2_vitg14")
    parser.add_argument("--imagenet_root", type=str, required=True,
                        help="ImageNet 경로 (train/, val/ 포함)")
    parser.add_argument("--norm_threshold", type=float, default=150.0)
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Linear probe 학습률 (기본: 0.1)")
    parser.add_argument("--probe_epochs", type=int, default=100,
                        help="Linear probe epoch 수 (기본: 100)")
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--train_subset", type=int, default=0,
                        help="Train set에서 사용할 이미지 수 (0=전체, 100000 권장)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모델 로드
    model = load_dinov2_model(args.model, device)
    
    # 데이터 로드
    transform = get_transform(224)
    
    train_dir = os.path.join(args.imagenet_root, "train")
    val_dir = os.path.join(args.imagenet_root, "val")
    
    print(f"\nImageNet train: {train_dir}")
    print(f"ImageNet val:   {val_dir}")
    
    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)
    print(f"Train (full): {len(train_ds)}, Val: {len(val_ds)}")
    
    # Subset 사용 시 클래스별 균등 샘플링
    if args.train_subset > 0 and args.train_subset < len(train_ds):
        from collections import defaultdict
        rng = np.random.RandomState(args.seed)
        
        # 클래스별 인덱스 수집
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(train_ds.samples):
            class_indices[label].append(idx)
        
        # 클래스별 균등 샘플링
        per_class = args.train_subset // len(class_indices)
        selected = []
        for label in sorted(class_indices.keys()):
            indices = class_indices[label]
            if len(indices) <= per_class:
                selected.extend(indices)
            else:
                selected.extend(rng.choice(indices, per_class, replace=False).tolist())
        
        train_ds = torch.utils.data.Subset(train_ds, selected)
        print(f"Train (subset): {len(train_ds)} ({per_class}/class × {len(class_indices)} classes)")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    
    # 여러 trial 실행
    results = {"CLS": [], "normal": [], "outlier": []}
    
    for trial in range(args.num_trials):
        seed = args.seed + trial
        print(f"\n{'='*60}")
        print(f"Trial {trial+1}/{args.num_trials} (seed={seed})")
        print(f"{'='*60}")
        
        # Train features
        print("\n[Train set]")
        train_cls, train_normal, train_outlier, train_labels, train_has_out = \
            extract_and_select(model, train_loader, device, args.norm_threshold, seed=seed)
        
        # Val features
        print("\n[Val set]")
        val_cls, val_normal, val_outlier, val_labels, val_has_out = \
            extract_and_select(model, val_loader, device, args.norm_threshold, seed=seed)
        
        # --- CLS linear probing ---
        if trial == 0:
            print("\n  Linear probing: [CLS]...")
            cls_acc = linear_probe(
                train_cls, train_labels,
                val_cls, val_labels,
                device=device, lr=args.lr, epochs=args.probe_epochs
            )
            print(f"  [CLS] Top-1 Acc: {cls_acc:.1f}%")
        results["CLS"].append(cls_acc)
        
        # --- Normal linear probing ---
        print("  Linear probing: [Normal]...")
        normal_acc = linear_probe(
            train_normal, train_labels,
            val_normal, val_labels,
            device=device, lr=args.lr, epochs=args.probe_epochs
        )
        print(f"  [Normal] Top-1 Acc: {normal_acc:.1f}%")
        results["normal"].append(normal_acc)
        
        # --- Outlier linear probing ---
        train_out_mask = train_has_out
        val_out_mask = val_has_out
        print(f"  Outlier 이미지: train={train_out_mask.sum()}/{len(train_out_mask)}, "
              f"val={val_out_mask.sum()}/{len(val_out_mask)}")
        
        if train_out_mask.sum() > 1000 and val_out_mask.sum() > 1000:
            print("  Linear probing: [Outlier]...")
            outlier_acc = linear_probe(
                train_outlier[train_out_mask], train_labels[train_out_mask],
                val_outlier[val_out_mask], val_labels[val_out_mask],
                device=device, lr=args.lr, epochs=args.probe_epochs
            )
            print(f"  [Outlier] Top-1 Acc: {outlier_acc:.1f}%")
            results["outlier"].append(outlier_acc)
        else:
            print("  [Outlier] 충분한 outlier 없음")
        
        # 메모리 정리
        del train_cls, train_normal, train_outlier
        del val_cls, val_normal, val_outlier
        torch.cuda.empty_cache()
    
    # 최종 결과
    print(f"\n\n{'='*60}")
    print(f"ImageNet-1k 최종 결과")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Norm threshold: {args.norm_threshold}")
    
    for token_type in ["CLS", "normal", "outlier"]:
        accs = results[token_type]
        if accs:
            mean = np.mean(accs)
            std = np.std(accs)
            print(f"  [{token_type:>7}] {mean:.1f} ± {std:.1f}%")
    
    print(f"\n논문 기준 (DINOv2-g):")
    print(f"  [CLS]     86.0%")
    print(f"  [normal]  65.8%")
    print(f"  [outlier] 69.0%")
    
    # 저장
    save_path = os.path.join(args.output_dir, f"imagenet_{args.model}.json")
    with open(save_path, "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2)
    print(f"\n결과 저장: {save_path}")


if __name__ == "__main__":
    main()