# 82.52% IDRiD 모델 재추론: QWK와 Class 1 전체 t-SNE

`scripts/idrid_seen_class1_tsne.py`는 기존 `hybrid_idrid_all_data_gem_no_spatial_seed42/final.pt`를 읽고 공식 test 103장을 다시 추론합니다. 모델은 test를 포함해 학습한 기존 모델이며, 재학습 없이 체크포인트의 전처리와 모델 설정을 사용합니다.

WSL/Bash에서 저장소 루트로 이동한 뒤 실행합니다.

```bash
git pull --ff-only origin learnable_dist
python scripts/idrid_seen_class1_tsne.py \
  --run-dir checkpoints/hybrid_idrid_all_data_gem_no_spatial_seed42 \
  --data-dir data/idrid \
  --device cpu \
  --output-dir outputs/idrid_8252_class1_all
```

CPU가 기본값이며 `--device cuda` 또는 `--device auto`도 지원합니다. 추론은 eval 모드, 학습 증강 없이, AMP를 끄고 수행합니다. 기존 평가 정확도와 재추론 정확도 차이를 함께 출력합니다. 기존 `predictions.csv`는 사용하지 않습니다. 기존 체크포인트·평가 파일·prepared 데이터는 읽기만 하며, 출력 폴더는 새 경로를 사용합니다.

정확도, 매크로 정확도, QWK, MAE와 클래스별 지표는 **test 전체**에서 계산합니다. t-SNE에는 정답 Class 1인 표본은 예측 결과와 관계없이 모두 넣고, Class 0·2·3·4는 정답 표본만 넣습니다. 이 선택은 PCA/t-SNE 적합 전에 적용합니다. **C1 오분류를 포함해 모든 점은 EyePACS와 같은 원형으로 표시**하며, 색상은 정답 클래스를 나타냅니다. 그림 제목의 `C1: all; C0/2/3/4: correct`와 CSV의 `correct` 열에 포함 규칙과 정답 여부를 보존합니다.

기존 82.52% 예측이 재현되면 Class 0/1/2/3/4에서 각각 31/5/28/14/12장, 총 90장을 그립니다. 실제 선택은 새 추론 결과에 따라 결정합니다. 특징은 각각 L2 정규화한 PCOL·SCOL 투영 벡터를 연결하고, t-SNE seed는 42입니다.

색상은 이전 EyePACS t-SNE와 동일하게 C0 파랑(`#1f77b4`), C1 주황(`#ff7f0e`), C2 초록(`#2ca02c`), C3 빨강(`#d62728`), C4 보라(`#9467bd`)를 사용합니다. 클래스가 그림에 빠져도 나머지 클래스의 색은 바뀌지 않습니다. 원래 EyePACS, IDRiD 정답 표본 전용 그림과 Class 1 전체 그림에서 같은 색상표를 공유합니다.

기본 계산·표시 설정은 EyePACS의 `save_outer_test_tsne`와 맞춥니다. 같은 특징과 표본을 넣었을 때 두 함수의 t-SNE 좌표가 일치하는 회귀 검사를 포함합니다.

| 항목 | 공통 설정 |
|---|---|
| 특징 | 각각 L2 정규화한 PCOL·SCOL 연결 |
| 추가 스케일링 | `none` |
| PCA | 최대 50차원, `svd_solver=auto`, `random_state=42` |
| t-SNE | `barnes_hut`, angle 0.5, perplexity 30, init PCA |
| 최적화 | 1000 iteration, learning rate `auto`, early exaggeration 12, seed 42 |
| 점·범례 | 모두 원형, 크기 12, alpha 0.7, `Class 0…4`, 범례 제목 `True label` |
| 그림 | 9×7인치, PNG 200 dpi, 벡터 PDF, 원래 EyePACS와 같은 자동 축 비율 |

표본 수가 적으면 PCA 차원과 perplexity를 원래 EyePACS 코드와 같은 규칙으로 제한합니다. 현재 IDRiD의 C1 전체/나머지 정답 표본 선택은 유지합니다. `--sweep`에서는 위 설정을 유지하면서 스케일링과 perplexity만 12가지 조합으로 바꿉니다.

출력 파일:

- `evaluation_metrics.json`: test 전체 지표, 체크포인트 SHA256·epoch, 이전 정확도와의 차이
- `per_class_metrics.csv`: test 전체 클래스별 지표
- `test_predictions.csv`: 재추론한 test 전체 예측
- `test_embeddings.npz`: test 전체 특징과 t-SNE 선택 여부
- `tsne_class1_all_other_correct.png` / `.pdf`: EyePACS와 같은 9×7인치, PNG 200 dpi, PDF 벡터 출력
- `tsne_coordinates.csv`: 선택된 영상의 정답·예측·정답 여부·좌표
- `tsne_metadata.json`: 선택 기준과 클래스별 포함 표본 수

출력 폴더가 이미 사용 중이면 `--output-dir outputs/idrid_8252_class1_all_run02`처럼 새 경로를 지정합니다.

## 기존 좌표를 유지하고 색상만 바꾸기

이미 그린 그림의 점 위치와 포함 표본을 유지하려면 저장된 좌표 CSV를 사용합니다. 다음 명령은 재학습·재추론이나 PCA/t-SNE 계산을 하지 않습니다.

```bash
python scripts/recolor_tsne.py \
  --coordinates outputs/idrid_8252_tsne_scaled/tsne_coordinates.csv \
  --output-dir outputs/idrid_8252_tsne_original_colors \
  --title 'IDRiD test (seen in training)'
```

`tsne_recolored.png`와 `.pdf`를 만들며, 좌표 원본은 `source_coordinates.csv`에 그대로 복사합니다. 기존 `tsne_test_correct_only.csv`나 `tsne_outer_test.csv`도 사용할 수 있습니다. `recolor_metadata.json`에는 원래 좌표 CSV의 경로·해시와 색상표를 기록합니다. 색상 변경은 모델의 예측이나 정확도·QWK를 변경하지 않습니다.

## 저장 특징의 스케일을 바꿔 다시 그리기

기존 분석 폴더의 `test_embeddings.npz`에는 test 전체 특징과 예측이 있습니다. 아래 명령은 영상·체크포인트를 읽거나 추론하지 않고 이 파일만 사용합니다.

```bash
python scripts/idrid_seen_class1_tsne.py \
  --embeddings outputs/idrid_8252_class1_all/test_embeddings.npz \
  --scaling none --perplexity 30 --max-iter 1000 \
  --output-dir outputs/idrid_8252_tsne_scaled --sweep
```

`--scaling`은 `none`(기존 특징), `standard`(각 특징 차원의 평균·분산 표준화), `whiten`(PCA 성분의 분산을 맞추는 whitening)을 지원합니다. Whitening에서는 최대 성분 분산의 1e-8 이하인 성분을 제외해 수치적으로 작은 성분의 확대를 제한합니다. 모든 변환은 Class 1 전체 + 나머지 정답 표본을 먼저 선택한 뒤 라벨을 사용하지 않고 계산합니다. 표준화와 whitening은 원래 거리 구조를 바꾸는 서로 다른 전처리이므로 설정을 그림과 함께 기록합니다.

`--perplexity`, `--learning-rate`, `--early-exaggeration`, `--pca-components`, `--max-iter`, `--point-size`와 seed 42로 재현 가능한 보기를 만들 수 있습니다. scikit-learn의 `n_iter`/`max_iter` 버전 차이를 처리합니다. 요청 perplexity가 표본 수 이상이면 표본 수보다 작게 제한하고 요청값과 실제값을 모두 기록합니다.

`--sweep`는 세 가지 scaling × perplexity 5/10/20/30의 12개 보기를 모두 저장합니다. `sweep/tsne_sweep.png` 또는 `.pdf`가 한눈에 보는 비교표이고, 각 하위 폴더에는 개별 PNG·벡터 PDF·좌표·설정이 있습니다. `tsne_input.npz`에는 선택된 원래 특징과 PCA 이후 특징을 함께 저장합니다. 원래 `evaluation_metrics.json`과 QWK·예측은 수정하지 않습니다.

모든 IDRiD t-SNE와 sweep은 EyePACS와 같은 **`method="barnes_hut"`**을 사용합니다. 이전 IDRiD 코드에서 사용한 `exact`는 제거했습니다. 실제 계산 방식은 각 `tsne_metadata.json`의 `method`에 기록합니다. BS가 학습 중인 경우 별도 터미널에서 다음 명령으로 기존 비-BS 모델의 12가지 변형을 CPU에서 생성합니다.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python scripts/idrid_seen_class1_tsne.py \
  --run-dir checkpoints/hybrid_idrid_all_data_gem_no_spatial_seed42 \
  --data-dir data/idrid --device cpu \
  --scaling none --perplexity 30 --max-iter 1000 --sweep \
  --output-dir "outputs/idrid_tsne_eyepacs_style_$(date +%Y%m%d_%H%M%S)"
```

축만 확대하면 겹치는 점의 관계는 바뀌지 않습니다. t-SNE 설정에 따라 그림이 달라질 수 있고 Class 0·1의 표현이 실제로 유사하면 겹침이 유지될 수 있습니다. 참고: [scikit-learn TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), [PCA whitening](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
