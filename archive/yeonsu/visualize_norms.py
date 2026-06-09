"""
Norm 분포 시각화 - Figure 3 재현
패치 토큰의 L2 norm 분포를 시각화하여 bimodal distribution 확인
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def collect_norms(model, dataloader, device="cuda", max_images=2000):
    """patch token norms 수집 - prenorm 기준"""
    all_norms = []
    count = 0
    
    for images, _ in tqdm(dataloader, desc="Collecting norms"):
        if count >= max_images:
            break
        images = images.to(device)
        output = model.forward_features(images)
        prenorm = output['x_prenorm']
        patch_tokens = prenorm[:, 1:]      # CLS 제외, LayerNorm 이전
        norms = patch_tokens.norm(dim=-1)  # [B, N]
        all_norms.append(norms.cpu())
        count += images.shape[0]
    
    return torch.cat(all_norms, dim=0)  # [total_images, N_patches]


def plot_norm_distribution(norms_dict, save_path="norm_distribution.png"):
    """
    여러 모델의 norm 분포를 한 그림에 시각화
    Figure 3 스타일
    """
    n_models = len(norms_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, norms) in zip(axes, norms_dict.items()):
        flat_norms = norms.flatten().numpy()
        
        # 히스토그램 (log scale)
        ax.hist(flat_norms, bins=200, density=True, alpha=0.7, 
                color='steelblue', edgecolor='none')
        ax.set_yscale('log')
        ax.set_xlabel('L2 norm', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{model_name} norms', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 통계 표시
        mean_norm = flat_norms.mean()
        outlier_ratio = (flat_norms > 150).mean() * 100
        ax.axvline(x=150, color='red', linestyle='--', alpha=0.7, label=f'threshold=150')
        ax.text(0.95, 0.95, 
                f'Mean: {mean_norm:.1f}\nOutlier(>150): {outlier_ratio:.2f}%',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_norm_map(model, image_tensor, device="cuda", save_path="norm_map.png"):
    """
    단일 이미지에 대한 norm map 시각화 - Figure 3 좌측 재현
    """
    model.eval()
    with torch.no_grad():
        output = model.forward_features(image_tensor.unsqueeze(0).to(device))
        prenorm = output['x_prenorm']
        patch_tokens = prenorm[:, 1:]      # CLS 제외, LayerNorm 이전
        norms = patch_tokens.norm(dim=-1)  # [1, N]
    
    # reshape to 2D
    n_patches = norms.shape[1]
    h = w = int(n_patches ** 0.5)
    norm_map = norms[0].cpu().reshape(h, w).numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 원본 이미지 (denormalize)
    img = image_tensor.clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    
    axes[0].imshow(img)
    axes[0].set_title("Input Image", fontsize=14)
    axes[0].axis('off')
    
    # Norm map
    im = axes[1].imshow(norm_map, cmap='hot', interpolation='nearest')
    axes[1].set_title("Patch Token Norms", fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Norm 분포 시각화")
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["dinov2_vitl14"],
                        help="비교할 모델 목록")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_images", type=int, default=2000,
                        help="Norm 수집에 사용할 최대 이미지 수")
    parser.add_argument("--output_dir", type=str, default="./results")
    
    args = parser.parse_args()
    device = f"cuda:{args.gpu}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # CIFAR-10으로 간단하게 시각화 (자동 다운로드)
    transform = get_transform(224)
    dataset = datasets.CIFAR10(root=args.data_root, train=False, 
                                download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, 
                        shuffle=False, num_workers=4)
    
    norms_dict = {}
    
    for model_name in args.models:
        print(f"\n{'='*40}")
        print(f"Model: {model_name}")
        print(f"{'='*40}")
        
        model = torch.hub.load('facebookresearch/dinov2', model_name)
        model = model.to(device).eval()
        
        norms = collect_norms(model, loader, device=device, max_images=args.max_images)
        norms_dict[model_name] = norms
        
        # 개별 이미지 norm map
        sample_image, _ = dataset[0]
        plot_norm_map(model, sample_image, device=device,
                      save_path=os.path.join(args.output_dir, f"norm_map_{model_name}.png"))
        
        del model
        torch.cuda.empty_cache()
    
    # 모든 모델 비교 플롯
    plot_norm_distribution(norms_dict, 
                           save_path=os.path.join(args.output_dir, "norm_distribution.png"))


if __name__ == "__main__":
    main()