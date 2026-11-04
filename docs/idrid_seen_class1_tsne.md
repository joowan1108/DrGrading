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

정확도, 매크로 정확도, QWK, MAE와 클래스별 지표는 **test 전체**에서 계산합니다. t-SNE에는 정답 Class 1인 표본은 예측 결과와 관계없이 모두 넣고, Class 0·2·3·4는 정답 표본만 넣습니다. 이 선택은 PCA/t-SNE 적합 전에 적용합니다. 색상은 정답 클래스를 나타내며 Class 1 오분류는 ×로 표시합니다. 그림 제목에도 `C1: all; C0/2/3/4: correct`를 표시합니다.

기존 82.52% 예측이 재현되면 Class 0/1/2/3/4에서 각각 31/5/28/14/12장, 총 90장을 그립니다. 실제 선택은 새 추론 결과에 따라 결정합니다. 특징은 각각 L2 정규화한 PCOL·SCOL 투영 벡터를 연결하고, t-SNE seed는 42입니다.

출력 파일:

- `evaluation_metrics.json`: test 전체 지표, 체크포인트 SHA256·epoch, 이전 정확도와의 차이
- `per_class_metrics.csv`: test 전체 클래스별 지표
- `test_predictions.csv`: 재추론한 test 전체 예측
- `test_embeddings.npz`: test 전체 특징과 t-SNE 선택 여부
- `tsne_class1_all_other_correct.png` / `.pdf`: 폭 3.5인치, PNG 600 dpi, PDF 벡터 출력
- `tsne_coordinates.csv`: 선택된 영상의 정답·예측·정답 여부·좌표
- `tsne_metadata.json`: 선택 기준과 클래스별 포함 표본 수

출력 폴더가 이미 사용 중이면 `--output-dir outputs/idrid_8252_class1_all_run02`처럼 새 경로를 지정합니다.

## 저장 특징의 스케일을 바꿔 다시 그리기

기존 분석 폴더의 `test_embeddings.npz`에는 test 전체 특징과 예측이 있습니다. 아래 명령은 영상·체크포인트를 읽거나 추론하지 않고 이 파일만 사용합니다.

```bash
python scripts/idrid_seen_class1_tsne.py \
  --embeddings outputs/idrid_8252_class1_all/test_embeddings.npz \
  --scaling standard --perplexity 10 --max-iter 3000 \
  --output-dir outputs/idrid_8252_tsne_scaled --sweep
```

`--scaling`은 `none`(기존 특징), `standard`(각 특징 차원의 평균·분산 표준화), `whiten`(PCA 성분의 분산을 맞추는 whitening)을 지원합니다. Whitening에서는 최대 성분 분산의 1e-8 이하인 성분을 제외해 수치적으로 작은 성분의 확대를 제한합니다. 모든 변환은 Class 1 전체 + 나머지 정답 표본을 먼저 선택한 뒤 라벨을 사용하지 않고 계산합니다. 표준화와 whitening은 원래 거리 구조를 바꾸는 서로 다른 전처리이므로 설정을 그림과 함께 기록합니다.

`--perplexity`, `--learning-rate`, `--early-exaggeration`, `--pca-components`, `--max-iter`, `--point-size`와 seed 42로 재현 가능한 보기를 만들 수 있습니다. scikit-learn의 `n_iter`/`max_iter` 버전 차이를 처리합니다. 요청 perplexity가 표본 수 이상이면 표본 수보다 작게 제한하고 요청값과 실제값을 모두 기록합니다.

`--sweep`는 세 가지 scaling × perplexity 5/10/20/30의 12개 보기를 모두 저장합니다. `sweep/tsne_sweep.png` 또는 `.pdf`가 한눈에 보는 비교표이고, 각 하위 폴더에는 개별 PNG·벡터 PDF·좌표·설정이 있습니다. `tsne_input.npz`에는 선택된 원래 특징과 PCA 이후 특징을 함께 저장합니다. 원래 `evaluation_metrics.json`과 QWK·예측은 수정하지 않습니다.

축만 확대하면 겹치는 점의 관계는 바뀌지 않습니다. t-SNE 설정에 따라 그림이 달라질 수 있고 Class 0·1의 표현이 실제로 유사하면 겹침이 유지될 수 있습니다. 참고: [scikit-learn TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), [PCA whitening](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
