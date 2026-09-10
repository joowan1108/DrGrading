# EyePACS 실험 결과 요약

## 1. 자료 범위와 해석 기준

`checkpoints.tar.gz`에서 실험 파일을 추출하여 각 디렉터리의 `run_config.json`과 `best_metrics.json`을 연결해 정리하였다. 최상위 실험 디렉터리는 11개이며, `hybrid_eyepacs_efficientnet_v2_s_e1_e2_gem384/fold_0`의 하위 실행까지 포함하면 실행 설정은 총 12개이다. 이 가운데 `best_metrics.json`이 존재하는 실행은 11개이고, `hybrid_eyepacs_efficientnet_v2_s_softplus_squared_mmnp_spatial_cbam_01`에는 설정 파일만 존재한다.

압축 파일에는 `evaluation_metrics.json`이 포함되어 있지 않다. 따라서 아래 수치는 모두 최적 체크포인트가 선택된 시점의 **검증 세트 성능**이며, 외부 테스트 세트의 최종 성능으로 해석해서는 안 된다. 모든 `best_metrics.json`은 `val.rmse_loss`를 체크포인트 선택 기준으로 사용한다.

일부 이전 실행에는 `macro_accuracy`와 `quadratic_weighted_kappa`가 기록되어 있지 않다. 비교표의 Macro Accuracy는 모든 실행에 동일한 기준을 적용하기 위해 Class 0–4의 클래스별 정확도를 단순 평균하여 다시 계산하였다. QWK는 파일에 저장된 값만 제시한다.

## 2. 공통 실험 설정

대부분의 실험은 다음 조건을 공유한다.

- 데이터셋: EyePACS (`trainLabels.csv`, 5개 등급)
- Backbone: ImageNet 사전학습 EfficientNetV2-S
- Batch size: 24
- Seed: 42
- 환자 독립적 분할 및 stratified batch 사용
- Projection head: hidden dimension 1,280, output dimension 128
- Regression input: projection과 backbone feature의 결합 표현
- Optimizer: Adam, learning rate 0.001, weight decay 0.0001
- RMSE loss weight: 1.0
- 클래스 경계 학습 손실 weight: 2.0
- Early stopping patience: 13
- AMP 및 gradient clipping 사용

주요 차이는 입력 크기와 정규화, pooling, spatial attention, 대조 손실 가중치 α와 β, 그리고 학습 가능한 경계의 초기화·스케일 설정이다.

## 3. 실험별 설정

| ID | 실험 디렉터리 | Fold | 입력/정규화 | Pooling | Spatial attention | α / β | 특징 |
|---|---|---:|---|---|:---:|---:|---|
| A1 | `hybrid_eyepacs_efficientnet_v2_s_384_avgpool_no_spatial` | 10 | 384 / ImageNet | Average | X | 0.1 / 0.1 | 384 해상도 pooling 기준선 |
| A2 | `hybrid_eyepacs_efficientnet_v2_s_384_gem_no_spatial` | 10 | 384 / ImageNet | GeM | X | 0.1 / 0.1 | A1에서 pooling만 GeM으로 변경 |
| A3 | `hybrid_eyepacs_efficientnet_v2_s_384_gem_spatial_cloc110` | 10 | 384 / ImageNet | GeM | X | 0.5 / 0.1 | 비균일 초기 경계 `[0.385, 0.451, 0.253, 0.330]` |
| A4 | `hybrid_eyepacs_efficientnet_v2_s_384_gem_spatial_cloc110-2` | 10 | 384 / ImageNet | GeM | X | 1.0 / 0.5 | A3 대비 α·β 증가 |
| A5 | `hybrid_eyepacs_efficientnet_v2_s_384_gem_spatial_cloc110_3` | 10 | 384 / ImageNet | GeM | X | 0.5 / 0.5 | A3 대비 β 증가 |
| A6 | `hybrid_eyepacs_efficientnet_v2_s_e1_e2_gem384` | 10 | 384 / ImageNet | GeM | O | 0.1 / 0.1 | A2와 주요 설정이 같고 spatial attention 사용 |
| CV1 | `hybrid_eyepacs_efficientnet_v2_s_e1_e2_gem384/fold_0` | 5 | 384 / ImageNet | GeM | O | 0.1 / 0.1 | 5-fold 실행 중 fold 0 결과만 존재 |
| B1 | `hybrid_eyepacs_efficientnet_v2_s_scaled_baseline_plus_mmnp` | 10 | 300 / 없음 | 미기록 | 미기록 | 1.0 / 1.0 | sigmoid 계열 경계, 초기값 0.2 |
| B2 | `hybrid_eyepacs_efficientnet_v2_s_softplus_squared_mmnp` | 10 | 300 / 없음 | 미기록 | 미기록 | 1.0 / 1.0 | softplus 및 제곱 순서 거리 |
| B3 | `hybrid_eyepacs_efficientnet_v2_s_softplus_squared_mmnp_regularized` | 10 | 300 / 없음 | 미기록 | 미기록 | 0.5 / 0.5 | ordinal scale 0.1, 별도 경계 손실 scale 0.25 |
| B4 | `hybrid_eyepacs_efficientnet_v2_s_softplus_squared_mmnp_spatial_cbam` | 10 | 300 / 없음 | 미기록 | O | 0.5 / 0.5 | spatial attention 사용 |
| B5 | `hybrid_eyepacs_efficientnet_v2_s_softplus_squared_mmnp_spatial_cbam_01` | 10 | 300 / 없음 | 미기록 | O | 0.1 / 0.1 | 결과 파일 없음 |

`spatial`이라는 문자열이 포함된 A3–A5의 디렉터리명과 달리, 해당 `run_config.json`의 `model.spatial_attention`은 모두 `false`이다. 결과 해석에는 디렉터리명이 아니라 저장된 실행 설정을 기준으로 사용해야 한다.

표의 `미기록`은 해당 옵션이 `run_config.json`에 저장되어 있지 않다는 뜻이며, 특정 기본값이 적용되었다고 임의로 간주하지 않았다.

## 4. 검증 성능 비교

Acc, Macro, C0–C4 및 Minor는 백분율이다. `Minor`는 연구에서 소수 클래스로 지정한 Class 1, 3, 4의 정확도 평균이다. CV1은 검증 표본 수와 분할 방식이 다른 5-fold 실행이므로 10-fold 결과와 직접적인 우열 비교에서 제외하는 것이 타당하다.

| ID | Best epoch | Acc | Macro | QWK | MAE | RMSE | C0 | C1 | C2 | C3 | C4 | Minor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A6 | 70 | 81.86 | 58.21 | — | 0.2271 | 0.5450 | 93.26 | 45.02 | 53.18 | 54.68 | 44.93 | 48.21 |
| A3 | 52 | 81.63 | **59.12** | **0.8231** | **0.2258** | **0.5388** | 94.03 | 35.28 | 50.11 | **68.35** | **47.83** | **50.48** |
| A2 | 59 | 81.45 | 57.52 | 0.8107 | 0.2331 | 0.5502 | 94.01 | 36.15 | 49.89 | 60.43 | 47.10 | 47.89 |
| A5 | 64 | 81.44 | 58.63 | 0.8121 | 0.2321 | 0.5467 | 92.59 | 38.74 | **55.73** | 58.99 | 47.10 | 48.28 |
| CV1 | 45 | 80.87 | 60.20 | — | 0.2327 | 0.5450 | 93.07 | 40.67 | 46.85 | 70.39 | 50.00 | 53.69 |
| A1 | 53 | 79.59 | 54.37 | — | 0.2476 | 0.5618 | 92.94 | 36.15 | 43.95 | 61.15 | 37.68 | 44.99 |
| A4 | 44 | 79.35 | 54.50 | 0.7945 | 0.2519 | 0.5664 | 90.74 | 36.80 | 54.03 | 48.92 | 42.03 | 42.58 |
| B4 | 48 | 74.41 | 38.74 | — | 0.3397 | 0.6915 | 88.74 | 17.53 | 47.77 | 29.50 | 10.14 | 19.06 |
| B2 | 39 | 74.11 | 43.14 | — | 0.3509 | 0.6992 | 88.39 | 14.94 | 44.59 | 43.88 | 23.91 | 27.58 |
| B3 | 26 | 72.69 | 43.89 | — | 0.3506 | 0.6890 | 85.51 | 21.00 | 46.50 | 32.37 | 34.06 | 29.14 |
| B1 | 39 | 71.54 | 41.16 | — | 0.3580 | 0.6922 | 84.62 | 25.76 | 42.78 | 33.81 | 18.84 | 26.14 |
| B5 | — | — | — | — | — | — | — | — | — | — | — | — |

## 5. 주요 결과

### 5.1 384 해상도 실험군

A6가 81.86%로 가장 높은 전체 정확도를 보였다. 다만 QWK가 기록되지 않아 순서형 분류 성능까지 종합해 A3보다 우수하다고 단정할 수는 없다. A3는 전체 정확도 81.63%, Macro Accuracy 59.12%, QWK 0.8231, MAE 0.2258, RMSE 0.5388을 기록했다. QWK가 저장된 실행 중 가장 높고 MAE와 RMSE도 가장 낮으며, Class 3 정확도 68.35%와 Class 4 정확도 47.83%를 확보해 10-fold 설정의 소수 클래스 평균도 50.48%로 가장 높았다. 현재 저장된 지표를 종합하면 A3가 가장 균형적인 후보이다.

A2와 A1은 입력 크기, 정규화, attention, 손실 가중치가 같고 pooling만 다르므로 비교 조건이 가장 명확하다. Average Pooling을 GeM으로 바꾸었을 때 전체 정확도는 79.59%에서 81.45%로 1.86%p 증가했고, Macro Accuracy는 54.37%에서 57.52%로 3.15%p 증가했다. 특히 Class 2는 5.94%p, Class 4는 9.42%p 향상되었으며 MAE는 0.2476에서 0.2331로, RMSE는 0.5618에서 0.5502로 감소했다. 반면 Class 1은 동일하고 Class 3은 0.72%p 낮아졌다. 전체적으로는 GeM의 효과가 긍정적이지만 모든 소수 클래스가 동시에 개선된 것은 아니다.

A6와 A2는 spatial attention의 사용 여부를 제외한 주요 설정이 같다. Attention을 사용한 A6는 전체 정확도가 0.41%p, Macro Accuracy가 0.69%p 증가했고 Class 1이 8.87%p 개선되었다. 반면 Class 3은 5.76%p, Class 4는 2.17%p 감소했다. 따라서 spatial attention은 경증 클래스 구분에는 유리했지만 중증 클래스 전체의 일관된 개선으로 이어지지는 않았다.

A3–A5를 비교하면 α=0.5, β=0.1인 A3가 가장 낮은 RMSE와 MAE, 가장 높은 QWK 및 Class 3·4 성능을 보였다. β를 0.5로 증가한 A5는 Class 1과 Class 2가 각각 38.74%, 55.73%로 개선되었으나 Class 3은 58.99%로 감소했다. α=1.0, β=0.5인 A4는 전체 정확도와 소수 클래스 평균이 모두 낮아져, 이 범위에서는 대조 손실 가중치를 크게 설정하는 것이 유리하지 않았다.

### 5.2 300 해상도 초기 실험군

B1–B4의 전체 정확도는 71.54–74.41%, 소수 클래스 평균은 19.06–29.14%로 384 해상도 실험군보다 전반적으로 낮았다. 그러나 두 실험군은 입력 크기뿐 아니라 ImageNet 정규화, dropout, pooling 설정 기록 여부, 손실 가중치와 경계 초기화 방식까지 함께 다르다. 따라서 성능 차이를 해상도 하나의 효과로 해석할 수는 없다.

B2는 B1보다 전체 정확도와 Class 3·4 성능이 높아 softplus 기반의 제곱 순서 거리가 sigmoid 계열 기준선보다 일부 소수 클래스에 유리한 경향을 보였다. B3는 Class 4가 34.06%로 B1과 B2보다 높았지만 Class 3은 32.37%로 감소했다. B4는 전체 정확도는 74.41%로 해당 그룹에서 가장 높지만 Class 4가 10.14%에 그쳐, 높은 전체 정확도가 소수 클래스 성능을 보장하지 않는다는 점을 보여준다.

### 5.3 5-fold 실행 상태

5-fold 설정은 `hybrid_eyepacs_efficientnet_v2_s_e1_e2_gem384/fold_0`에서만 확인된다. Fold 0의 검증 성능은 전체 정확도 80.87%, Macro Accuracy 60.20%, Class 1·3·4 평균 53.69%이다. 그러나 fold 1–4의 결과와 전체 fold 집계 파일이 없으므로 현재 자료만으로 5-fold 교차검증의 평균과 표준편차를 산출할 수 없다. 상위 디렉터리의 `run_config.json`은 10-fold로 기록되어 있어 하위 fold 0 설정과도 일치하지 않는다.

## 6. 학습된 클래스 경계

아래 값은 `best_metrics.json`의 Class 0–1, 1–2, 2–3, 3–4 경계값 순서이다.

| ID | 0–1 | 1–2 | 2–3 | 3–4 | 정지 조건 |
|---|---:|---:|---:|---:|---|
| A1 | 0.8116 | 0.8242 | 0.5533 | 0.4851 | Phase 1 최대 epoch |
| A2 | 0.7852 | 0.7976 | 0.5333 | 0.4672 | Phase 1 최대 epoch |
| A3 | 0.2991 | 0.3565 | 0.1990 | 0.2617 | Phase 1 최대 epoch |
| A4 | 0.3470 | 0.4100 | 0.2295 | 0.2995 | 수렴 판정 |
| A5 | 0.2997 | 0.3585 | 0.2003 | 0.2608 | Phase 1 최대 epoch |
| A6 | 0.7653 | 0.4549 | 0.4954 | 0.6200 | Phase 1 최대 epoch |
| CV1 | 0.7367 | 0.4356 | 0.4748 | 0.5955 | 수렴 판정 |
| B1 | 0.1923 | 0.1977 | 0.1768 | 0.1677 | 수렴 판정 |
| B2 | 0.8094 | 0.8229 | 0.5519 | 0.4838 | Phase 1 최대 epoch |
| B3 | 0.8378 | 0.8723 | 0.5994 | 0.5073 | 수렴 판정 |
| B4 | 0.7451 | 0.4422 | 0.4813 | 0.6026 | 수렴 판정 |

학습된 네 경계는 대부분 동일한 값으로 수렴하지 않았다. 이는 인접 등급 간 관계를 일률적인 간격으로 고정하기보다 데이터에서 서로 다른 경계 구조를 학습할 수 있음을 보여준다. 다만 경계값의 절대 크기는 parameterization, 초기화와 정규화 방식에 따라 달라지므로 서로 다른 실험 설정 사이에서 직접 비교하기보다 동일 설정 안에서의 상대적 패턴을 해석해야 한다.

## 7. 결론 및 후속 확인 사항

현재 검증 결과에서는 384×384 입력, ImageNet 정규화 및 GeM pooling 조합이 300 해상도 초기 실험군보다 안정적인 성능을 보인다. 전체 정확도만 보면 A6가 가장 높지만, 저장된 순서형 지표와 소수 클래스 성능을 함께 고려하면 A3가 가장 균형적인 10-fold 후보이다. 특히 Class 3과 Class 4 개선에 강점이 있다. 반면 Class 1은 A6가 가장 높아, 소수 클래스별 목표에 따라 최적 설정이 달라질 수 있다.

최종 결론을 내리기 전에 다음 자료가 추가로 필요하다.

- 모든 실험의 `evaluation_metrics.json`
- 5-fold 실험의 fold 1–4 결과
- 5개 fold의 평균과 표준편차를 담은 집계 결과
- QWK가 누락된 실행의 재평가 결과
- 설정만 존재하는 B5의 완료 여부 확인

이 자료가 확보되면 검증 성능이 아니라 외부 테스트 성능을 기준으로 모델을 선정하고, fold 간 변동성까지 포함한 신뢰도 높은 비교가 가능하다.
