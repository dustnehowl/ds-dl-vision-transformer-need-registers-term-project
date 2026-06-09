# Vision Transformers Need Registers — Reproduction Study

> Darcet et al., "Vision Transformers Need Registers", ICLR 2024
> ([arXiv:2309.16588](https://arxiv.org/abs/2309.16588))

Term project for DS/DL course: reproducing key experiments from the paper using DINOv2 backbones.

---

## Overview

The paper identifies **artifact tokens** (high-norm outlier patches) in ViT feature maps and proposes appending **register tokens** during training to eliminate them. We reproduce three groups of experiments to verify the claims:

| Experiment | Paper Reference | What we verify |
|---|---|---|
| **Figure 3** | Norm distribution | Outlier patches show bimodal L2 norm distribution |
| **Table 1** | Token-level probing | Outlier tokens carry global (class) information |
| **Table 2** | Downstream tasks | Registers improve ImageNet classification, ADE20k segmentation, NYUd depth |

---

## Repository Structure

```
.
├── fig3_norm_visualization.py      # Figure 3: patch norm distribution & norm map
├── table1_token_probing.py         # Table 1: CLS / normal / outlier token linear probing
├── table2_imagenet_extract.py      # Table 2: ImageNet feature extraction (multi-GPU)
├── table2_imagenet_linear.py       # Table 2: ImageNet linear classification
├── table2_ade20k_segmentation.py   # Table 2: ADE20k linear segmentation (4-layer BNHead)
├── table2_nyud_depth.py            # Table 2: NYUd monocular depth (Official BNHead)
├── results/                        # Experiment result JSONs
│   ├── table1_vitg14.json
│   ├── table2_imagenet.json
│   ├── table2_ade20k_noreg.json
│   ├── table2_ade20k_reg.json
│   └── table2_nyud.json
└── archive/                        # Previous exploration notebooks & scripts
```

---

## Experiments & Results

### Figure 3: Patch Token Norm Distribution

Patch token의 L2 norm (LayerNorm 이전, `x_prenorm`)을 시각화하여 outlier의 bimodal distribution을 확인합니다.

```bash
# DINOv2 ViT-L (without registers) — CIFAR-10 validation set
python fig3_norm_visualization.py --models dinov2_vitl14 --gpu 0

# With vs Without registers 비교
python fig3_norm_visualization.py --models dinov2_vitl14 dinov2_vitl14_reg --gpu 0
```

**Observation**: `dinov2_vitl14`에서 norm > 150인 outlier patch가 전체의 약 2-3%를 차지하며, register 모델에서는 이 bimodal peak가 사라짐.

---

### Table 1: Token-level Linear Probing

CLS token, normal patch, outlier patch 각각으로 linear probing하여 outlier token이 global information을 carry하는지 확인합니다.

```bash
# DINOv2 ViT-G14, auto threshold (상위 2.37%)
python table1_token_probing.py \
    --model dinov2_vitg14 \
    --datasets CIFAR10 CIFAR100 Aircraft DTD Flowers102 Food101 Pets Caltech101 CUB200 \
    --auto_threshold \
    --gpu 0
```

**Results (DINOv2 ViT-G14, 224px)**:

| Dataset | CLS | Normal Patch | Outlier Patch | Delta (Outlier - Normal) |
|---|---:|---:|---:|---:|
| CIFAR10 | 99.46 | 97.42 | 99.23 | +1.81 |
| CIFAR100 | 93.95 | 82.16 | 92.70 | +10.54 |
| Food101 | 94.81 | 76.42 | 92.90 | +16.48 |
| CUB200 | 91.28 | 19.58 | 84.62 | +65.04 |
| Aircraft | 87.25 | 18.83 | 74.66 | +55.83 |
| Caltech101 | 93.31 | 74.24 | 96.36 | +22.12 |
| Flowers102 | 99.69 | 61.60 | 99.60 | +38.00 |
| Pets | 96.59 | 50.59 | 93.96 | +43.37 |
| DTD | 81.81 | 58.48 | 83.28 | +24.80 |

**Key finding**: 모든 데이터셋에서 outlier patch의 accuracy가 normal patch보다 크게 높음 → outlier token이 class-level global information을 담고 있음을 확인 (논문의 Table 1 trend 재현 성공).

---

### Table 2: Downstream Task Performance

Register token 추가에 따른 downstream task 성능 변화를 재현합니다.

#### (a) ImageNet Linear Classification

DINOv2 best config (4 blocks CLS concat + avgpool → 5120-dim feature)으로 linear probing.

```bash
# Step 1: Feature extraction (multi-GPU)
python table2_imagenet_extract.py --n_gpus 5 --imagenet_root /path/to/imagenet

# Step 2: Linear classifier training
python table2_imagenet_linear.py --gpu 0 --feature_dir ./features_phase1
```

| Model | Top-1 Accuracy (%) | Paper |
|---|---:|---:|
| DINOv2 ViT-L14 | 86.04 | 84.3 |
| DINOv2 ViT-L14 + reg | 86.66 | 84.8 |
| **Delta** | **+0.62** | **+0.5** |

> Our reproduced values are higher than the paper (likely due to different hyperparameters), but the **positive delta (+0.62 vs +0.5)** is consistent.

#### (b) ADE20k Semantic Segmentation

Frozen backbone + 4-layer BNHead (BatchNorm → Conv2d 1x1) linear segmentation.

```bash
python table2_ade20k_segmentation.py \
    --ade20k_root /path/to/ADEChallengeData2016 \
    --models dinov2_vitl14 dinov2_vitl14_reg \
    --image_size 518 --n_iter 20000 --gpu 0
```

| Model | mIoU (%) | Paper |
|---|---:|---:|
| DINOv2 ViT-L14 | 48.96 | 46.6 |
| DINOv2 ViT-L14 + reg | 50.31 | 47.9 |
| **Delta** | **+1.35** | **+1.3** |

> Delta (+1.35 vs +1.3) matches the paper closely.

#### (c) NYUd Monocular Depth Estimation

Official BNHead protocol: 4-layer features with CLS broadcast, UD bins, SigLoss.

```bash
python table2_nyud_depth.py \
    --mat_path /path/to/nyu_depth_v2_labeled.mat \
    --train_split /path/to/train.txt --test_split /path/to/test.txt \
    --models dinov2_vitl14 dinov2_vitl14_reg \
    --n_iter 38400 --gpu 0
```

| Model | RMSE (m) | Paper |
|---|---:|---:|
| DINOv2 ViT-L14 | 0.4957 | 0.378 |
| DINOv2 ViT-L14 + reg | 0.4711 | 0.366 |
| **Delta** | **-0.0246** | **-0.012** |

> Absolute RMSE is higher than paper (training data/augmentation differences), but the **improvement direction with registers is consistent** (lower RMSE with reg).

---

## Environment

- **GPU**: NVIDIA A6000 (48GB) x 8
- **Framework**: PyTorch 2.x + torchvision
- **Backbone**: DINOv2 (via `torch.hub.load('facebookresearch/dinov2', ...)`)
- **Datasets**: ImageNet-1K, ADE20K (ADEChallengeData2016), NYU Depth v2, CIFAR-10/100, CUB-200, FGVCAircraft, DTD, Flowers102, Food101, OxfordIIITPet, SUN397, Caltech101

### Requirements

```bash
pip install torch torchvision numpy matplotlib tqdm h5py pillow
```

---

## Team

| Member | Contribution |
|---|---|
| Yeonsu Kim | Table 1 (token probing), Table 2 (ImageNet, ADE20k, NYUd), Figure 3, experiment pipeline |
| Eunjung | DINOv1 norm visualization |
| Hyunbin | SigLIP artifact analysis, cosine similarity experiments |

---

## References

- Darcet, T., Oquab, M., Mairal, J., & Jegou, H. (2024). *Vision Transformers Need Registers*. ICLR 2024.
- Oquab, M., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. arXiv:2304.07193.
- Chen, T., et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020.
