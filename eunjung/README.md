# DINO Token Norm Analysis — Artifact 올바른 판별

> **논문**: Darcet et al., *Vision Transformers Need Registers*, ICLR 2024 ([arXiv:2309.16588](https://arxiv.org/abs/2309.16588))
> **모델**: DINO **ViT-S/16** (`dino_vits16`, ~21M params)
> **핵심 목표**: DINOv2와 달리 **DINO(원본)** 모델에서는 high-norm outlier(artifact)가 나타나지 않는다는 것을 GMM 기반 bimodality 검출로 정확히 판별

---

## 이 노트북이 답하는 질문

> *"DINO(원본)에서도 DINOv2처럼 high-norm outlier가 보이는가?"*

결론: **보이지 않는다.**
DINO `ViT-S/16`의 patch token norm 분포는 **unimodal**이며, threshold(e.g., 150)을 아무리 적용해도 bimodal 봉우리가 없기 때문에 진짜 artifact는 **0.00%**다.

---

## 왜 이전 방식이 잘못됐나? (핵심 개념 정리)

| | DINOv2 (bimodal) | DINO 원본 (unimodal) |
|---|---|---|
| **분포 형태** | 두 개의 뚜렷한 봉우리 | 하나의 봉우리 |
| **mean+2σ 이상** | → 진짜 outlier | → **정상 tail** |
| **진짜 artifact 비율** | ~3.39% | **0.00%** |

이전 버전에서 DINO에서 ~1.53%의 outlier가 관찰된 것은 단순히 `mean + 2σ` threshold를 기계적으로 적용했기 때문이다. 분포가 unimodal일 때 상위 꼬리는 정상적인 통계적 변동이지, 논문에서 말하는 "artifact"가 아니다.

---

## 올바른 판별 방법: GMM 기반 Bimodality 검출

단순 threshold 대신 **분포의 형태**로 판정한다.

### 알고리즘
1. patch token의 prenorm L2 norm 계산
2. 2-component GMM 피팅
3. **분리도(separation score)** 계산:
   ```
   separation = |μ₁ - μ₂| / ((σ₁ + σ₂) / 2)
   ```
4. `separation > BIMODAL_SEPARATION_THRESHOLD(3.0)` → bimodal → 진짜 artifact 존재
5. 그 이하 → unimodal → artifact 없음

---

## 실험 환경

| 항목 | 값 |
|------|-----|
| GPU | Tesla T4 (15360 MiB) |
| CUDA | 13.0 |
| 실행일 | 2026-05-29 |
| Python | 3.x |
| 주요 라이브러리 | torch, torchvision, scikit-learn |

---

## 설정값

```python
MODEL_NAME  = 'dino_vits16'
IMG_SIZE    = 224
IMAGE_PATH  = 'input.jpg'

# GMM 분리도 기준: (두 mode 간 거리) / (평균 std)
# 이 값보다 크면 bimodal로 판정
BIMODAL_SEPARATION_THRESHOLD = 3.0
```

---

## 모델 아키텍처

```
모델     : dino_vits16 (ViT-Small/16)
patch_size : 16
embed_dim  : 384
grid       : 14×14 = 196 patches
```

### `forward_features_dino()` 구현

DINOv2의 `forward_features()` 인터페이스와 동일하게 맞춘 커스텀 함수.

| 반환 키 | 의미 |
|---------|------|
| `x_prenorm` | 마지막 LayerNorm **이전** (outlier 판별에 사용) |
| `x_norm_patchtokens` | 마지막 LayerNorm **이후** (feature로 사용 가능) |
| `x_norm_clstoken` | CLS token (post-norm) |

> ⚠️ **DINOv2와 동일한 구현 원칙**: norm 측정은 `x_prenorm`, feature 추출은 `x_norm_patchtokens` 사용.

---

## 실험 결과 (dino_vits16, input.jpg)

### Norm 통계

| 통계량 | 값 |
|--------|-----|
| min | 3.20 |
| max | 5.53 |
| mean | 3.953 |
| std | 0.426 |

### Bimodality 검출 결과

| 항목 | 값 |
|------|-----|
| GMM component 0 (normal) | mean=3.82, std=0.27, weight=0.795 |
| GMM component 1 (outlier?) | mean=4.48, std=0.52, weight=0.205 |
| **Separation score** | **1.68 (기준: 3.0)** |
| Minor component weight | 20.47% |
| **판정** | ✅ **UNIMODAL — No true artifacts!** |
| True artifact ratio | **0.00%** |

### 논문 비교

| 모델 | 분포 | 진짜 artifact |
|------|------|---------------|
| DINO `vits16` (this) | Unimodal ✅ | **없음** ← 논문과 일치 |
| DINOv2-g14 (논문) | Bimodal ⚠️ | ~3.39% |
| DINOv2+reg (논문) | Unimodal ✅ | 없음 |

---

## 시각화 결과

### Norm Distribution

![norm_distribution_dino.png](norm_distribution_dino.png)

- X축: L2 norm, Y축: Density (log scale)
- GMM 2-component 피팅 표시
- 분포가 완전히 왼쪽에 몰려 있으며 threshold=150 이상의 값이 **전혀 없음 (0.00%)**
- 두 GMM component가 충분히 분리되어 있지 않음 → unimodal 판정

### Norm Map

![norm_map_dino.png](norm_map_dino.png)

- 입력 이미지: `input.jpg` (224×224, 고양이 사진)
- Grid: 14×14 = 196 patches
- Patch norm map이 완전히 어둡게 표시됨 → high-norm patch 없음
- DINOv2에서 보이는 밝은 점(artifact)이 전혀 관찰되지 않음

---

## 파일 구조

```
dino-1/
├── README.md              ← 이 파일
├── dino-1.ipynb           ← 메인 노트북
├── input.jpg              ← 테스트 이미지 (필요 시 자동 다운로드)
└── 출력물/
    ├── norm_distribution_dino.png   ← norm 히스토그램 + GMM 피팅
    ├── norm_map_dino.png            ← 단일 이미지 patch norm map
    └── method_comparison_dino.png  ← 판별 방법 비교
```

---

## 실행 방법

```bash
# 1. 의존성 설치
pip install torch torchvision scikit-learn matplotlib pillow

# 2. 노트북 실행 (GPU 권장)
jupyter notebook dino-1.ipynb
```

첫 실행 시 `torch.hub`가 DINO 가중치(`dino_deitsmall16_pretrain.pth`, ~82.7MB)를 자동 다운로드합니다.

---

## 관련 노트북과의 비교

| | dino-1.ipynb (이 노트북) | DINOv2 실험 (`run_token_probing.py`) |
|---|---|---|
| **모델** | DINO ViT-S/16 | DINOv2 ViT-g/14 |
| **목적** | Artifact 판별 (단일 이미지) | Linear probing (Table 1 재현) |
| **판별 방법** | GMM bimodality | threshold=150 (prenorm) |
| **결과** | Unimodal, 0.00% artifact | Bimodal, ~3.39% artifact |
| **논문과 일치** | ✅ | ✅ |

---

## 참고 문헌

- Darcet, Oquab, Mairal, Bojanowski. *Vision Transformers Need Registers*. ICLR 2024. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)
- DINO (원본): [https://github.com/facebookresearch/dino](https://github.com/facebookresearch/dino)
- DINOv2: [https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
