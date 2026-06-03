# Hyunbin — Artifact Similarity & SigLIP Analysis

> **논문**: Darcet et al., *Vision Transformers Need Registers*, ICLR 2024
>
> **실험 목표**
>
> 1. Artifact token이 주변 patch와 높은 cosine similarity를 가지는지 검증
> 2. Register가 없는 Vision-Language ViT(SigLIP)에서도 artifact 현상이 관찰되는지 확인

---

## 실험 1 — Cosine Similarity Analysis

논문 Figure 5(a)에 따르면 artifact patch는 인접 patch와 매우 높은 cosine similarity를 보인다.

이를 검증하기 위해 DINOv2 ViT-g/14를 사용하여:

* patch embedding에서 cosine similarity 계산
* final output token norm으로 artifact 판별
* artifact patch와 normal patch의 cosine similarity 분포 비교

### 실행 방법

```bash
RUN_NAME=dinov2_g_val50k_thr150_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 python cos_similarity_exp.py \
  --model dinov2_vitg14 \
  --data_dir /data/imagenet_1k/val \
  --num_images 50000 \
  --batch_size 1 \
  --threshold 150 \
  --output_dir ./results/$RUN_NAME
```

### 결과

Artifact patch의 평균 cosine similarity는 normal patch보다 높게 나타났다.

| Token Type     | Mean Cosine Similarity |
| -------------- | ---------------------- |
| Normal Patch   | 0.620                  |
| Artifact Patch | 0.823                  |

이는 artifact token이 주변 patch와 매우 유사한 정보를 반복적으로 저장한다는 논문의 주장과 일치한다.

![cosine similarity](results/cos_similarity.png)

---

## 실험 2 — SigLIP Artifact Analysis

논문은 self-supervised ViT(DINOv2)에서 artifact 현상을 보고하였다.

본 실험에서는 Vision-Language 모델인 SigLIP에서도 동일한 현상이 관찰되는지 분석하였다.

분석 방법:

* final layer patch token norm 측정
* high-norm patch 탐색
* layer-wise norm map 시각화
* dataset-level norm distribution 분석

### 실행 방법

#### Dataset-level norm distribution

```bash
python siglip_dataset_norm_fig3.py
```

#### Single-image artifact visualization

```bash
python siglip_artifact_probe_full.py \
  --image /path/to/image.jpg \
  --model google/siglip-base-patch16-224 \
  --manual_threshold 150 \
  --outdir ./outputs_siglip_visual_one
```

### 결과

SigLIP에서도 일부 patch token이 높은 norm을 가지는 현상을 확인하였다.

다만 DINOv2와 동일한 artifact 메커니즘인지 여부는 추가 분석이 필요하다.

![siglip histogram](results/siglip_histogram.png)

---

## 참고

* Darcet et al., Vision Transformers Need Registers, ICLR 2024
* DINOv2
* SigLIP
