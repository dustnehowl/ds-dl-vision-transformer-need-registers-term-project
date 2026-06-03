# Hyunbin — Cosine Similarity & SigLIP Artifact Analysis

> **논문**: Darcet et al., *Vision Transformers Need Registers*, ICLR 2024
> **목표**: *Vision Transformers Need Registers* 논문의 artifact token 분석을 재현하고 추가 실험을 수행한다.

본 실험은 논문에서 제시한 **high-norm artifact token** 현상을 두 방향에서 분석한다.

1. DINOv2에서 artifact token이 주변 patch와 높은 cosine similarity를 가지는지 확인
2. Vision-Language ViT 계열인 SigLIP에서도 유사한 high-norm artifact가 관찰되는지 확인

---

## File Structure

```bash
hyunbin/
├── README.md
├── scripts/
│   ├── cos_similarity_exp.py
│   ├── replot_cos.py
│   ├── siglip_artifact_probe_full.py
│   └── siglip_dataset_norm_fig3.py
├── cos_sim_results.png
├── siglip_artifact_bimodal.png
├── siglip_artifact_example.png
└── siglip_artifact_layer.png
```

---

# 1. Cosine Similarity Analysis

## Motivation

논문 Figure 5(a)는 **high-norm artifact token이 주변 patch token과 매우 높은 cosine similarity를 가진다**고 보고한다.

이는 artifact token이 새로운 시각 정보를 담는 것이 아니라, 주변 patch에 이미 존재하는 **중복적인 local information**을 저장하거나 흡수하는 방식으로 사용될 수 있음을 의미한다.

따라서 본 실험에서는 DINOv2 ViT-g/14를 사용하여 다음 질문을 확인하였다.

> Artifact token은 실제로 normal patch token보다 주변 patch와 더 유사한가?

---

## Method

실험은 ImageNet validation set 50,000장을 대상으로 수행하였다.

* Model: `dinov2_vitg14`
* Dataset: ImageNet validation set
* Number of images: 50,000
* Artifact threshold: patch token norm > 150
* Similarity target: 4-neighbor patches
* Similarity feature: patch embedding output

실험 절차는 다음과 같다.

1. ViT 최종 출력에서 patch token의 L2 norm을 계산한다.
2. norm이 150보다 큰 patch token을 artifact token으로 정의한다.
3. artifact token의 위치를 기준으로 patch embedding layer 직후의 feature를 가져온다.
4. 해당 patch와 상하좌우 4-neighbor patch 사이의 cosine similarity를 계산한다.
5. normal patch와 artifact patch의 cosine similarity distribution을 비교한다.

여기서 cosine similarity는 최종 transformer output이 아니라 **patch embedding 직후 feature**에서 계산하였다.
최종 출력은 이미 transformer block을 여러 번 통과하여 global information이 섞여 있기 때문에, 해당 위치가 원래 이미지에서 중복적인 local 영역이었는지를 확인하기 위해 초기 patch embedding feature를 사용하였다.

---

## Run

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

---

## Results

```text
===== Result Summary =====
elapsed sec: 2849.91
normal cos count: 46809338
outlier cos count: 1190662
norm min / mean / max: 25.404 / 59.588 / 582.710
threshold mean: 150.000
normal cos mean: 0.6204
outlier cos mean: 0.8228
```

| Token Type       | Mean Cosine Similarity |
| ---------------- | ---------------------- |
| Normal patches   | 0.6204                 |
| Artifact patches | 0.8228                 |

### Cosine Similarity Distribution

Artifact patch는 일반 patch보다 주변 4개 이웃 patch와 훨씬 높은 cosine similarity를 보였다.
또한 artifact 분포는 cosine similarity가 1.0에 가까운 영역에 강하게 집중되어 있었으며, 이는 논문 Figure 5(a)의 결과와 일치한다.

이 결과는 high-norm artifact token이 균일한 배경과 같이 patch 정보가 중복되는 영역에서 주로 발생한다는 것을 시사한다.
즉, 이러한 token은 고유한 지역 시각 정보를 표현하기보다는 중복 정보나 전역 정보를 저장하는 내부 저장소 역할을 수행할 가능성이 있다.

Outlier 비율은 다음과 같다.

```text
1,190,662 / 48,000,000 ≈ 2.48%
```

이는 논문에서 보고된 DINOv2의 high-norm token 비율인 약 2.37%와 매우 유사하다.

---

# 2. SigLIP Artifact Analysis

## Motivation

원 논문에서는 high-norm artifact token 현상이 주로 DINOv2에서 관찰되며, 기존 DINO에서는 동일한 패턴이 나타나지 않는다고 보고한다.

하지만 SigLIP 역시 ViT 기반 구조를 사용하는 모델이다.
따라서 본 실험에서는 다음 질문을 확인하고자 하였다.

> SigLIP에서도 DINOv2와 유사한 high-norm patch-token artifact가 나타나는가?

이 실험은 논문의 DINOv2 재현을 넘어서는 추가 분석에 해당한다.

---

## 데이터셋 단위 Norm 분석

SigLIP-B/16 모델을 사용하여 데이터셋 전체에 대한 norm 분석을 수행하였다.

* Model: `google/siglip-base-patch16-224`
* 입력 해상도: 224 × 224
* 데이터셋 크기: 50,000장
* Artifact 기준: patch token norm > 150

전체 데이터셋에 대해 마지막 레이어의 patch-token norm을 수집하였다.

```text
processed this run: 50000
global max this run: 255.3673
total_patch_tokens: 9800000
mean_estimated_from_hist: 46.4388
median_estimated_from_hist: 43.0
p99_estimated_from_hist: 154.0
p99_9_estimated_from_hist: 235.0
threshold: 150.0
outlier_count: 98775
outlier_ratio_percent: 1.0079
```

### Dataset-Level Norm Distribution

히스토그램을 보면 대부분의 patch token은 norm 값이 25~50 범위에 분포하지만, 일부 token은 190~245 부근의 별도 고-norm 영역을 형성한다.
이는 SigLIP 역시 이봉 분포(bimodal distribution)에 가까운 norm 분포를 보인다는 것을 의미한다. 다만 outlier 비율은 DINOv2보다 더 낮게 나타났다.

---

## 단일 이미지 Artifact 시각화

Artifact 패턴을 정성적으로 확인하기 위해 가장 높은 artifact score를 가진 이미지를 선택하였다.

Artifact score는 다음과 같이 정의하였다.

```text
artifact score = max patch token norm / median patch token norm
```

```bash
python siglip_artifact_probe_full.py \
  --image /path/to/image.jpg \
  --model google/siglip-base-patch16-224 \
  --manual_threshold 150 \
  --outdir ./outputs_siglip_visual_one
```

### Example Artifact Visualization

선택된 이미지에서는 소수의 patch token만 threshold를 초과하였다.
SigLIP-B/16은 224×224 입력에서 14×14 patch grid를 생성하므로 이미지당 총 196개의 patch token이 존재한다.
약 1.5% 수준의 outlier 비율은 실제로는 약 3개의 patch에 해당한다.

이는 high-norm 현상이 이미지 전체에 퍼져 있는 것이 아니라 매우 적은 수의 patch에 국소적으로 집중되어 나타난다는 것을 보여준다.

---

## 레이어별 분석

레이어별 norm 히스토그램과 norm map도 함께 시각화하였다.

### Layer-wise Norm Evolution

레이어별 분석 결과는 다음과 같은 특징을 보였다.

* 초기 레이어에서는 norm 값이 상대적으로 작고 뚜렷한 high-norm artifact가 나타나지 않는다.
* 레이어가 깊어질수록 최대 norm 값이 점진적으로 증가한다.
* 마지막 레이어에 가까워질수록 소수의 token이 매우 큰 norm 값을 갖게 된다.
* high-norm token은 공간적으로 특정 위치에 집중되어 나타난다.

이러한 현상은 논문에서 설명한 artifact 형성 과정과 유사하다. 즉, high-norm token은 처음부터 존재하는 것이 아니라 깊은 레이어를 거치면서 점차 형성되는 것으로 보인다.

---

# Conclusion

Cosine similarity 실험은 high-norm artifact token이 중복적인 patch 영역에서 주로 발생한다는 논문의 주장을 뒷받침한다.

DINOv2 ViT-g/14의 결과는 다음과 같다.

* Artifact patch는 일반 patch보다 주변 patch와 더 높은 유사도를 가진다.
* Artifact cosine mean: 0.8228
* Normal cosine mean: 0.6204
* Outlier 비율: 약 2.48%

또한 SigLIP 실험을 통해 Vision-Language ViT 모델에서도 high-norm patch token이 나타날 수 있음을 확인하였다.

SigLIP-B/16의 결과는 다음과 같다.

* 대부분의 patch token은 norm 25~50 범위에 분포한다.
* 일부 token은 threshold 150을 초과하는 고-norm 그룹을 형성한다.
* 데이터셋 전체 기준 outlier 비율은 약 1.01%이다.
* 레이어가 깊어질수록 high-norm token이 형성된다.
* Artifact 영역은 공간적으로 매우 국소적으로 나타난다.

따라서 SigLIP 역시 artifact와 유사한 high-norm patch token 현상을 보이는 것으로 판단된다.
다만 이러한 token이 DINOv2의 register artifact와 동일한 메커니즘이나 역할을 수행하는지는 추가적인 분석이 필요하다.

---

## Notes

* 대용량 데이터셋 및 `.npz` 결과 파일은 저장소에 포함하지 않았다.
* 코드 스크립트와 대표 결과 이미지들만 업로드하였다.
* ImageNet 데이터셋은 별도로 준비해야 한다.
