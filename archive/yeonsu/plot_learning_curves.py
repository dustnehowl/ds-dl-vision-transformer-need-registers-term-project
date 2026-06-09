"""
Linear Probe 학습 곡선 시각화

- normal 토큰: 데이터/epoch이 많아질수록 천천히 성능 향상
- outlier 토큰: 비교적 빠르게 수렴
- CLS 토큰: 기준선
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import json


def get_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_features(model, dataloader, device, norm_threshold):
    """CLS / normal / outlier token 추출"""
    rng = np.random.RandomState(42)
    
    all_cls, all_normal, all_outlier, all_labels = [], [], [], []
    has_outlier = []
    
    for images, targets in tqdm(dataloader, desc="Extracting"):
        images = images.to(device)
        output = model.forward_features(images)
        
        prenorm = output['x_prenorm']
        norms = prenorm[:, 1:].norm(dim=-1)       # prenorm으로 outlier 판별
        cls_tokens = output['x_norm_clstoken']     # post-norm feature
        patch_tokens = output['x_norm_patchtokens'] # post-norm feature
        
        for i in range(images.shape[0]):
            img_norms = norms[i]
            img_patches = patch_tokens[i]
            
            normal_idx = (img_norms <= norm_threshold).nonzero(as_tuple=True)[0]
            outlier_idx = (img_norms > norm_threshold).nonzero(as_tuple=True)[0]
            
            all_cls.append(cls_tokens[i].cpu())
            all_labels.append(targets[i].item())
            
            if len(normal_idx) > 0:
                idx = normal_idx[rng.randint(len(normal_idx))]
                all_normal.append(img_patches[idx].cpu())
            else:
                all_normal.append(torch.zeros(img_patches.shape[1]))
            
            if len(outlier_idx) > 0:
                idx = outlier_idx[rng.randint(len(outlier_idx))]
                all_outlier.append(img_patches[idx].cpu())
                has_outlier.append(True)
            else:
                all_outlier.append(torch.zeros(img_patches.shape[1]))
                has_outlier.append(False)
    
    return (torch.stack(all_cls),
            torch.stack(all_normal),
            torch.stack(all_outlier),
            np.array(all_labels),
            np.array(has_outlier))


def train_and_record(train_feat, train_labels, test_feat, test_labels,
                     device, lr=0.1, epochs=100, batch_size=1024,
                     weight_decay=1e-4, record_every=1):
    """
    Linear probe 학습하면서 epoch마다 val accuracy 기록
    Returns: epoch별 accuracy list
    """
    # numpy/tensor 처리
    def to_tensor(x, dtype):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype)
        return x.to(dtype)
    
    train_feat = to_tensor(train_feat, torch.float32)
    train_labels = to_tensor(train_labels, torch.long)
    test_feat = to_tensor(test_feat, torch.float32)
    test_labels = to_tensor(test_labels, torch.long)
    
    # 정규화
    mean = train_feat.mean(dim=0)
    std = train_feat.std(dim=0).clamp(min=1e-6)
    train_feat = (train_feat - mean) / std
    test_feat = (test_feat - mean) / std
    
    num_classes = train_labels.max().item() + 1
    feat_dim = train_feat.shape[1]
    
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    nn.init.zeros_(classifier.bias)
    nn.init.normal_(classifier.weight, std=0.01)
    
    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(
        TensorDataset(train_feat, train_labels),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    # 평가 함수
    def evaluate():
        classifier.eval()
        with torch.no_grad():
            correct = 0
            test_loader = DataLoader(
                TensorDataset(test_feat, test_labels),
                batch_size=4096, shuffle=False
            )
            for f, l in test_loader:
                f, l = f.to(device), l.to(device)
                correct += (classifier(f).argmax(1) == l).sum().item()
        classifier.train()
        return correct / len(test_labels) * 100
    
    # 학습 + 기록
    acc_curve = []
    for epoch in range(epochs):
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            loss = criterion(classifier(feats), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % record_every == 0:
            acc_curve.append(evaluate())
    
    return acc_curve


def plot_curves(curves_dict, save_path, title="Linear Probe 학습 곡선"):
    """
    여러 토큰의 학습 곡선을 한 그래프에 시각화
    curves_dict: {"CLS": [...], "normal": [...], "outlier": [...]}
    """
    colors = {"CLS": "gray", "normal": "steelblue", "outlier": "tomato"}
    linestyles = {"CLS": "--", "normal": "-", "outlier": "-"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for token_type, accs in curves_dict.items():
        epochs = list(range(1, len(accs) + 1))
        ax.plot(epochs, accs,
                color=colors[token_type],
                linestyle=linestyles[token_type],
                linewidth=2,
                label=f"{token_type} (final: {accs[-1]:.1f}%)")
    
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Val Accuracy (%)", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(list(curves_dict.values())[0]))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dinov2_vitg14")
    parser.add_argument("--dataset", type=str, default="Aircraft",
                        choices=["Aircraft", "DTD", "CIFAR100",
                                 "Flowers102", "Food101", "Pets"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--norm_threshold", type=float, default=150.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--record_every", type=int, default=1,
                        help="몇 epoch마다 accuracy 기록 (기본: 1)")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모델 로드
    print(f"Loading {args.model}...")
    model = torch.hub.load('facebookresearch/dinov2', args.model)
    model = model.to(device).eval()
    
    # 데이터 로드
    transform = get_transform()
    
    if args.dataset == "Aircraft":
        train_ds = datasets.FGVCAircraft(args.data_root, split="train",
                                          download=True, transform=transform)
        test_ds = datasets.FGVCAircraft(args.data_root, split="test",
                                         download=True, transform=transform)
    elif args.dataset == "DTD":
        train_ds = datasets.DTD(args.data_root, split="train",
                                 download=True, transform=transform)
        test_ds = datasets.DTD(args.data_root, split="test",
                                download=True, transform=transform)
    elif args.dataset == "CIFAR100":
        train_ds = datasets.CIFAR100(args.data_root, train=True,
                                      download=True, transform=transform)
        test_ds = datasets.CIFAR100(args.data_root, train=False,
                                     download=True, transform=transform)
    elif args.dataset == "Flowers102":
        train_ds = datasets.Flowers102(args.data_root, split="train",
                                        download=True, transform=transform)
        test_ds = datasets.Flowers102(args.data_root, split="test",
                                       download=True, transform=transform)
    elif args.dataset == "Food101":
        train_ds = datasets.Food101(args.data_root, split="train",
                                     download=True, transform=transform)
        test_ds = datasets.Food101(args.data_root, split="test",
                                    download=True, transform=transform)
    elif args.dataset == "Pets":
        train_ds = datasets.OxfordIIITPet(args.data_root, split="trainval",
                                           download=True, transform=transform)
        test_ds = datasets.OxfordIIITPet(args.data_root, split="test",
                                          download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
    
    # Feature 추출
    print("\n[Train] Feature extraction...")
    train_cls, train_normal, train_outlier, train_labels, train_has_out = \
        extract_features(model, train_loader, device, args.norm_threshold)
    
    print("[Test] Feature extraction...")
    test_cls, test_normal, test_outlier, test_labels, test_has_out = \
        extract_features(model, test_loader, device, args.norm_threshold)
    
    outlier_ratio = train_has_out.mean() * 100
    print(f"\nOutlier 비율: {outlier_ratio:.2f}%")
    print(f"Outlier 있는 train 이미지: {train_has_out.sum()}/{len(train_has_out)}")
    
    # 학습 곡선 기록
    curves = {}
    
    print(f"\n[CLS] Linear probe ({args.epochs} epochs)...")
    curves["CLS"] = train_and_record(
        train_cls, train_labels, test_cls, test_labels,
        device=device, lr=args.lr, epochs=args.epochs,
        record_every=args.record_every
    )
    print(f"  Final: {curves['CLS'][-1]:.1f}%")
    
    print(f"\n[Normal] Linear probe ({args.epochs} epochs)...")
    curves["normal"] = train_and_record(
        train_normal, train_labels, test_normal, test_labels,
        device=device, lr=args.lr, epochs=args.epochs,
        record_every=args.record_every
    )
    print(f"  Final: {curves['normal'][-1]:.1f}%")
    
    if train_has_out.sum() > 100:
        print(f"\n[Outlier] Linear probe ({args.epochs} epochs)...")
        curves["outlier"] = train_and_record(
            train_outlier[train_has_out], train_labels[train_has_out],
            test_outlier[test_has_out], test_labels[test_has_out],
            device=device, lr=args.lr, epochs=args.epochs,
            record_every=args.record_every
        )
        print(f"  Final: {curves['outlier'][-1]:.1f}%")
    
    # 시각화
    save_path = os.path.join(
        args.output_dir,
        f"learning_curve_{args.model}_{args.dataset}.png"
    )
    plot_curves(
        curves, save_path,
        title=f"Linear Probe 학습 곡선 ({args.model}, {args.dataset})"
    )
    
    # 결과 저장
    json_path = save_path.replace(".png", ".json")
    with open(json_path, "w") as f:
        json.dump({"model": args.model, "dataset": args.dataset,
                   "curves": curves}, f, indent=2)
    print(f"Data saved: {json_path}")


if __name__ == "__main__":
    main()