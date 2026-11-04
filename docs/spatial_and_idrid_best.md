# Spatial Attention 맵과 지정 모델의 IDRiD 비교 실험

Spatial Attention에는 사용자가 선택한 `DrGrading/checkpoints/hybrid_eyepacs_efficientnet_v2_s_softplus_squared_mmnp_spatial_cbam/best.pt`를 사용합니다. 이 실험은 입력 300×300, `[0,1]` 전처리, Average Pooling, Spatial Attention 적용 설정입니다. IDRiD에는 `hybrid_eyepacs_efficientnet_v2_s_384_gem_no_spatial`의 저장된 설정을 사용하며, 입력 384×384와 GeM을 사용하고 Balanced Softmax와 Spatial Attention은 적용하지 않습니다. 맵 추출용 체크포인트는 IDRiD 학습 가중치나 설정으로 전달하지 않습니다.

`hybrid_eyepacs_efficientnet_v2_s_384_gem_spatial_cloc110-2`는 이름과 달리 저장된 `run_config.json`에서 `spatial_attention: false`인 실험입니다. 이 체크포인트로는 학습된 Spatial Attention 맵을 추출할 수 없습니다. `checkpoints.tar.gz`의 원래 실행 기록과 `checkpoint_experiment_summary.md`에서도 이 설정을 확인할 수 있습니다. 원격에 존재하는 실제 가중치는 다음 명령으로 확인합니다.

```bash
python scripts/export_spatial_attention.py --list-runs
```

이 명령은 현재 프로젝트와 옆 프로젝트의 체크포인트 폴더를 찾아 `best.pt`의 실제 공간 주의 가중치와 저장 설정을 확인합니다. `READY`인 항목에는 절대 경로로 된 실행 명령을 출력합니다. `BEST MISSING`은 설정 파일은 있지만 실제 `best.pt`가 없는 상태이고, `NO SPATIAL WEIGHTS`는 체크포인트에 공간 주의 가중치가 없는 상태입니다. 다른 위치를 검색하려면 `--search-root /실제/경로/checkpoints`를 추가합니다.

## 한 번에 실행

원격 저장소의 루트에서 실행합니다. 이전에 만든 `data/idrid/prepared/manifest.json`, `train.csv`, `test.csv`를 재사용합니다. 데이터를 준비하거나 내려받는 작업은 실행하지 않습니다.

```bash
git pull --ff-only origin learnable_dist
python scripts/run_spatial_and_idrid_best.py \
  --spatial-run ../DrGrading/checkpoints/hybrid_eyepacs_efficientnet_v2_s_softplus_squared_mmnp_spatial_cbam \
  --source-run checkpoints/hybrid_eyepacs_efficientnet_v2_s_384_gem_no_spatial \
  --data-dir data/idrid \
  --device cuda \
  --output-dir outputs/spatial_and_idrid_best_seed42 \
  --skip-completed
```

먼저 EyePACS 검증 세트에서 **정답이 Class 3인 한 장**을 추론하여 실제 공간 가중치를 원본 안저 영상에 중첩합니다. 이어서 IDRiD의 두 모델을 각각 새로 학습하고 최종 평가와 t-SNE를 저장합니다.

위 경로는 `DrGrading-idrid`와 `DrGrading`이 같은 상위 폴더에 있을 때의 예시입니다. 기본 실험명도 위 CBAM 실험으로 지정되어 있으므로, `--spatial-run`을 생략하면 같은 이름의 폴더를 현재·옆 프로젝트에서 찾습니다. `--skip-completed`는 같은 설정으로 평가까지 끝난 IDRiD 실험을 재학습하지 않고 재사용합니다.

Spatial Attention 폴더는 현재 checkout과 옆 프로젝트들의 `checkpoints/`에서 정확한 이름으로 찾습니다. 위치가 다르거나 같은 이름의 폴더가 여러 개면 `--spatial-run` 대신 `--spatial-checkpoint /실제/경로/best.pt`를 사용합니다. 저장된 EyePACS 영상 경로를 바꾸려면 `--eyepacs-labels-csv /경로/trainLabels.csv --eyepacs-image-root /경로/train`을 추가합니다. 기존 IDRiD 준비 파일이 다른 곳에 있으면 `--data-dir`만 바꿉니다.

## 두 IDRiD 실험

| 출력 폴더 | 학습 표본 | 평가 표본 | test 정확도의 의미 |
|---|---|---|---|
| `train_plus_test_seen` | 공식 train 413장 + test 103장 | 공식 test 103장 전체 | 학습에 포함한 test 영상을 다시 평가한 정확도 |
| `train_only_heldout` | 공식 train 413장 | 공식 test 103장 전체 | 이 실행의 학습에서 제외한 test 정확도 |

두 실험은 같은 seed 42와 초기화, 학습 횟수를 사용합니다. `train_only_heldout`에서는 test와 파일 내용 또는 디코딩된 픽셀이 동일한 train 영상이 있으면 그 **train 복사본만 제외**합니다. 공식 test 103장은 유지하고 실제 제외 목록을 저장합니다. 환자 정보가 제공되지 않는 경우 파일·픽셀 검사만으로 환자 단위 독립성까지 확인할 수는 없습니다.

각 모델은 ImageNet 초기 가중치에서 시작합니다. EyePACS 체크포인트의 가중치를 IDRiD로 옮기지 않고 **설정만 재사용**합니다. 기본값은 저장소에 기록된 해당 EyePACS 실행의 설정을 고정한 `configs/eyepacs_best_gem_no_spatial_source.json`이며, 실제 원격 실행 폴더의 저장 설정을 직접 읽으려면 다음 옵션을 추가합니다.

```bash
  --source-run checkpoints/hybrid_eyepacs_efficientnet_v2_s_384_gem_no_spatial
```

| 설정 | 적용값 |
|---|---|
| 모델·입력 | EfficientNetV2-S, 384×384, GeM, Spatial Attention 없음, Balanced Softmax 없음 |
| 영상 전처리·증강 | 원래 EyePACS의 검은 테두리 제거, bicubic resize, 수평 뒤집기, ±10° 회전, 밝기·대비·채도 변형, ImageNet 정규화 |
| PCOL / SCOL / RMSE / margin 손실 가중치 | 0.1 / 0.1 / 1 / 2 |
| MMNP 집계 | 표본 쌍에 대해 합산, anchor에 대해 평균: 원래 EyePACS 실행과 동일 |
| 학습률 | 네트워크 `1e-3`, margin `1e-5` |
| Phase 1 | 최소 20, 최대 30 epoch; 원래 간격 수렴 기준 유지 |
| 총 학습 | 75 epoch, batch size 24, 두 실험 모두 epoch당 22회 업데이트: 각각 1,650회 |
| 클래스 표본 추출 | 각 미니배치에 클래스별 최소 2장 포함 |
| IDRiD에 맞춘 선택·스케줄링 | 내부 검증 없음, train RMSE 기반 scheduler, early stopping 없음, 마지막 epoch의 `final.pt` 평가 |

이전 IDRiD 스크립트의 224 입력, margin 학습률 `1e-3`, Phase 1 50–100 epoch와 pair-mean 설정은 사용하지 않습니다. `--epochs 500`처럼 학습 기간을 명시하면 **두 실험 모두** 같은 기간을 적용합니다. 총 epoch는 Phase 1 최대 기간보다 길어야 합니다. `--batches-per-epoch`로 두 실험의 업데이트 횟수를 함께 바꿀 수 있습니다. test 지표로 scheduler, 종료 시점 또는 체크포인트를 고르지 않습니다.

## 결과 파일

기본 출력 루트는 `outputs/spatial_and_idrid_best_seed42/`입니다.

```text
spatial_class3/
  overlay_original.png             # 원본 영상 위 Spatial Attention
  comparison.png / comparison.pdf   # 입력·공간 가중치·중첩 비교
  spatial_attention.npy            # 실제 sigmoid 공간 가중치
  metadata.json                    # 실제 체크포인트·영상·정답·예측 정보
idrid_comparison/
  comparison.csv                   # 두 실험의 전체·클래스별 결과 한 표
  comparison.json                  # 두 실험 상세 결과와 완료 여부
  source_eyepacs_config.json        # 사용한 원래 EyePACS 설정과 출처
  train_plus_test_seen/
  train_only_heldout/
    final.pt
    run_config.json
    training_split.json            # 실제 학습·test ID, 정답, 픽셀 해시
    evaluation_metrics.json
    training_pool/                 # 최종 모델로 학습 표본 전체를 다시 평가
      predictions.csv
      per_class_metrics.csv
      evaluation_metrics.json
    test/                          # 공식 test 전체 평가, 양쪽 실험에서 같은 파일 구성
      predictions.csv
      confusion_matrix.csv
      per_class_metrics.csv
      evaluation_metrics.json
      tsne_test_correct_only.png
      tsne_test_correct_only.pdf
      tsne_test_correct_only.csv
      tsne_test_correct_only_embeddings.npz
      tsne_test_correct_only_metadata.json
```

`comparison.csv`에는 학습 풀 정확도와 공식 test 정확도, 매크로 정확도, QWK, MAE, Class 0–4 정확도 및 표본 수를 기록합니다. 클래스별 정확도는 **해당 정답 클래스에서 올바르게 분류한 비율(recall)**입니다. 정확도·매크로 정확도·클래스별 정확도의 CSV 열은 `%` 단위입니다. MAE는 연속 예측값을 0–4로 반올림·제한한 등급과 정답 등급의 차이로 계산합니다.

지표는 test 전체에서 계산합니다. t-SNE는 **정답과 예측 등급이 일치하는 test 표본만 먼저 추린 뒤** PCA와 t-SNE를 계산합니다. 특징은 각각 L2 정규화한 PCOL·SCOL 투영 벡터의 연결이며, EyePACS와 같은 9×7인치·200 dpi PNG와 벡터 PDF를 저장합니다. 정답 표본이 3개 미만이거나 특징이 모두 동일하면 좌표를 만들지 않고 사유를 기록합니다. `training-seen test`와 `held-out test`를 그림 제목과 메타데이터에도 구분합니다.

## 각각 실행하기

Spatial Attention만 추출하려면:

```bash
python scripts/export_spatial_attention.py \
  --run-dir ../DrGrading/checkpoints/hybrid_eyepacs_efficientnet_v2_s_softplus_squared_mmnp_spatial_cbam \
  --class-id 3 --split val --device cuda \
  --output-dir outputs/spatial_attention_class3
```

IDRiD 두 실험만 실행하려면:

```bash
python scripts/compare_idrid_best_eyepacs.py \
  --data-dir data/idrid --device cuda \
  --output-dir checkpoints/idrid_best_eyepacs_comparison_seed42
```

하나만 실행할 때는 `--protocol train_only_heldout` 또는 `--protocol train_plus_test_seen`을 추가합니다. 결합 실행에서는 `--skip-spatial`로 맵 추출을 건너뛸 수 있습니다. 다른 Class 3 영상은 `--spatial-sample-index 1`로 선택합니다.

앞선 실행이 Spatial Attention 단계에서 실패했다면 IDRiD 학습은 아직 시작되지 않은 상태입니다. IDRiD부터 실행하려면 기존 명령에 `--skip-spatial`을 추가합니다.

```bash
python scripts/run_spatial_and_idrid_best.py \
  --skip-spatial \
  --source-run checkpoints/hybrid_eyepacs_efficientnet_v2_s_384_gem_no_spatial \
  --data-dir data/idrid --device cuda \
  --output-dir outputs/spatial_and_idrid_best_seed42
```

출력 디렉터리는 새 경로를 사용합니다. 같은 설정으로 일부 실험이 **평가까지 완료된** 경우 `--skip-completed`를 추가하면 완료된 실험을 재사용하고 나머지 실험을 실행합니다. 미완료 학습을 이어서 훈련하는 옵션은 아니며, 중단된 학습은 새 출력 경로에서 시작해야 합니다.
