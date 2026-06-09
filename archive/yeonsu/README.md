# Yeonsu — DINOv2 Outlier Token Linear Probing 재현

> **논문**: Darcet et al., *Vision Transformers Need Registers*, ICLR 2024 ([arXiv:2309.16588](https://arxiv.org/abs/2309.16588))
> **재현 목표**: **Table 1** — CLS / normal patch / outlier patch token의 linear probing 성능 비교
> **모델**: DINOv2 **ViT-g/14** (`dinov2_vitg14`, 1.1B params)

사전학습된 DINOv2 ViT-g/14를 그대로 가져와서, patch token 중 약 3%를 차지하는 **high-norm
outlier(artifact) token**이 *local 정보를 버리고 global 정보를 담는다*는 논문의 주장을
linear probing으로 검증한다.

- **CLS token**: 원래 global 정보를 담는 토큰 (기준선)
- **normal patch token** (norm ≤ threshold): local 정보 중심
- **outlier patch token** (norm > threshold): 논문 주장대로라면 global 정보 보유

→ image classification(=global task)에서 **outlier ≫ normal** 이면 논문 주장이 지지된다.

---

## 파일 구조

```
yeonsu/
├── README.md
├── requirements.txt
├── run_token_probing.py        # Table 1 재현: 소규모 데이터셋 token별 linear probing (sklearn)
├── run_imagenet_probing.py     # ImageNet-1k용 메모리 효율 버전 (PyTorch GPU SGD probe)
├── plot_learning_curves.py     # token별 linear probe 학습 곡선
├── visualize_norms.py          # norm 분포 히스토그램 + norm map 시각화
└── results/
    ├── results_dinov2_vitg14.json          # ViT-g/14 token probing 결과
    ├── imagenet_dinov2_vitg14.json         # ImageNet-1k token probing 결과
    ├── learning_curve_dinov2_vitg14_Aircraft.json / .png
    ├── norm_distribution.png               # patch norm 히스토그램 (bimodal)
    └── norm_map_dinov2_vitg14.png          # 단일 이미지 patch norm map (artifact 위치)
```

> 데이터셋(`data/`, 약 19GB)은 용량 문제로 포함하지 않는다. CIFAR/Flowers/Pets/Aircraft/DTD는
> `torchvision`이 `--data_root` 경로에 자동 다운로드하며, ImageNet은 직접 준비해야 한다.

---

## 실행 환경

- Python 3.10+, CUDA GPU 1장 (RTX 4090 환경 기준)
- DINOv2는 `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')`로 자동 다운로드

```bash
pip install -r requirements.txt
```

---

## 핵심 구현 이슈 — prenorm vs post-norm (삽질 기록)

이 재현에서 가장 오래 헤맸던 부분. DINOv2의 `forward_features()` 출력에는 **norm 스케일이
완전히 다른 두 가지 표현**이 들어 있다.

| 출력 키 | 의미 | patch norm 범위 | outlier 관찰 |
|---------|------|-----------------|--------------|
| `x_norm_patchtokens` | 마지막 LayerNorm **이후** (post-norm) | 약 **40~60** (좁음) | ❌ 안 보임 |
| `x_prenorm` | 마지막 LayerNorm **이전** (pre-norm) | 약 **28~545** (bimodal) | ✅ ~3% 관찰 |

논문은 norm을 측정한 대상을 그냥 *"output of the model"* 이라고만 적어 두고 prenorm/post-norm을
명시하지 않는다. 그래서 처음엔 자연스럽게 `x_norm_patchtokens`(post-norm)로 norm을 계산했는데,
**LayerNorm이 norm을 40~60의 좁은 범위로 압축해 버려 outlier가 전혀 보이지 않는** 문제가 있었다.
threshold를 아무리 바꿔도 bimodal 분포가 나오지 않아 한참 헤맸다.

**올바른 구현** (본 레포의 모든 스크립트가 따르는 방식):

- **norm 계산 / outlier 판별** → `x_prenorm` 사용 (LayerNorm 이전이라야 high-norm artifact가 드러남)
- **linear probing feature** → `x_norm_patchtokens` 사용 (post-norm이 분류 학습에 더 안정적)

즉, *"어떤 토큰이 outlier인가"는 prenorm으로 정하고, 그 토큰의 feature는 post-norm 값을 쓴다.*

아래 norm 분포가 이 차이를 그대로 보여 준다 — prenorm 기준에서만 좌(normal)·우(outlier)
**bimodal 봉우리**가 나타나고, threshold=150 기준 outlier 비율은 **3.39%** 로 논문의 2.37%와 근사하다.

![norm distribution](results/norm_distribution.png)

단일 이미지의 patch norm map에서도 배경 등 일부 패치에서만 norm이 폭발하는 artifact(밝은 점)가
또렷하게 보인다.

![norm map](results/norm_map_dinov2_vitg14.png)

---

## 실행 방법

```bash
# (1) norm 분포 / norm map 시각화
python visualize_norms.py --models dinov2_vitg14 --gpu 0 --max_images 2000

# (2) Table 1 재현 — token별 linear probing
python run_token_probing.py \
    --model dinov2_vitg14 \
    --datasets CIFAR10 CIFAR100 Flowers102 Pets Aircraft DTD \
    --norm_threshold 150 --num_trials 5 --gpu 0

# (3) ImageNet-1k (메모리 효율 버전, train subset 사용)
python run_imagenet_probing.py \
    --model dinov2_vitg14 --imagenet_root /path/to/imagenet \
    --train_subset 100000 --num_trials 3 --gpu 0

# (4) linear probe 학습 곡선
python plot_learning_curves.py --model dinov2_vitg14 --dataset Aircraft --epochs 200 --gpu 0
```

---

## 실험 결과 (Table 1 재현)

DINOv2 ViT-g/14, norm threshold = 150. 재현값은 random token 선택을 5회 반복한 평균±표준편차
(편차 표기가 없는 값은 deterministic). 단위는 Top-1 accuracy (%).

| Token | CIFAR-10 | CIFAR-100 | Flowers102 | Pets | Aircraft | DTD |
|-------|:--------:|:---------:|:----------:|:----:|:--------:|:---:|
| CLS (논문) | 99.4 | 94.5 | 99.7 | 96.9 | 87.3 | 85.2 |
| **CLS (재현)** | 99.5 | 94.0 | 99.7 | 96.4 | 72.7 | 82.0 |
| normal (논문) | 97.1 | 81.3 | 59.5 | 47.8 | 17.1 | 63.1 |
| **normal (재현)** | 97.3 ±0.1 | 81.9 ±0.1 | 35.5 ±1.3 | 49.7 ±0.8 | 12.6 ±0.6 | 58.6 ±0.6 |
| outlier (논문) | 99.3 | 93.7 | 99.6 | 94.1 | 79.1 | 84.9 |
| **outlier (재현)** | 99.3 | 93.6 | 99.4 | 94.5 ±0.2 | 70.5 ±0.2 | 83.4 ±0.1 |

핵심 경향은 모든 데이터셋에서 동일하게 재현된다: **outlier ≫ normal**, 그리고 outlier는 CLS에
근접한다. local 정보 중심인 normal patch는 fine-grained 데이터셋(Flowers102, Pets, Aircraft)에서
특히 크게 뒤처져, **outlier token이 global 정보를 담는다는 논문의 핵심 주장을 분명하게 재현**한다.

### ImageNet-1k (train 100k subset 사용 — 논문과 직접 비교 불가)

| Token | 논문 (full 1.28M) | 재현 (100k subset) |
|-------|:-----------------:|:------------------:|
| CLS | 86.0 | 82.7 |
| normal | 65.8 | 44.9 |
| outlier | 69.0 | 79.8 |

절대 수치는 train 규모(100k vs 1.28M)와 probe 설정 차이로 논문과 직접 비교할 수 없지만,
**outlier > normal** 패턴은 동일하게 확인된다.

---

## 특이사항 및 분석

- **CIFAR-10/100, Pets, DTD**: 논문 수치와 거의 일치 — 재현 성공.
- **Flowers102 normal**: 논문(59.5) 대비 크게 낮음(35.5). `torchvision`의 Flowers102 **train split이
  클래스당 10장(총 1,020장)** 밖에 안 되어 발생한 **dataset split 차이**로 추정.
- **Aircraft CLS**: 논문(87.3) 대비 낮음(72.7). 논문의 정확한 전처리/평가 세팅을 확인하지 못함
  (해상도·crop 등 차이 가능성).
- **ImageNet**: train subset(100k)을 사용해 논문(full 1.28M)과 **직접 비교 불가**. 다만 패턴
  (outlier > normal)은 동일하게 관찰됨.
- **Norm threshold 150**은 논문(DINOv2-g 기준)을 그대로 사용. 실측 outlier 비율은 **~3.0~3.5%** 로
  논문의 2.37%와 근사.

---

## 실험 재현 시 주의사항

1. **norm은 반드시 `x_prenorm`으로 측정한다.** post-norm(`x_norm_patchtokens`)으로 측정하면 norm이
   40~60으로 압축되어 outlier가 전혀 보이지 않는다. (위 *핵심 구현 이슈* 참고)
2. **norm threshold는 모델마다 다르다.** 150은 DINOv2 ViT-g/14 기준값이다. 다른 백본은 prenorm norm의
   절대 스케일이 달라지므로, `run_token_probing.py`의 `--auto_threshold`(상위 약 2.37% 자동 결정)를
   쓰거나 norm 분포를 먼저 확인하고 threshold를 조정해야 한다.
3. **데이터셋 split을 확인한다.** torchvision의 기본 split이 논문과 다를 수 있다(특히 Flowers102는
   train이 1,020장으로 매우 작음). split 차이는 normal/CLS 절대 수치에 직접 영향을 준다.
4. **ImageNet은 전용 스크립트를 쓴다.** 1.28M 이미지의 모든 patch token을 RAM에 올릴 수 없으므로,
   `run_imagenet_probing.py`는 추출과 동시에 이미지당 토큰 1개만 선택해 저장하고 GPU SGD로 probe한다.
   `--train_subset`으로 규모를 조절하되, 논문과 절대 수치를 비교하려면 full set이 필요하다.
5. **probe optimizer 차이를 감안한다.** 소규모 데이터셋은 sklearn `LogisticRegression`(LBFGS),
   ImageNet은 PyTorch SGD를 쓴다. optimizer·하이퍼파라미터에 따라 절대 정확도가 달라질 수 있어
   *경향(outlier vs normal)* 비교에 무게를 두는 것이 안전하다.
6. **GPU와 torch.hub 다운로드가 필요하다.** 첫 실행 시 DINOv2 가중치를 자동 내려받는다.

---

## 참고

- Darcet, Oquab, Mairal, Bojanowski. *Vision Transformers Need Registers*. ICLR 2024. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)
- DINOv2: <https://github.com/facebookresearch/dinov2>
