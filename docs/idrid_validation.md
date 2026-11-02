# IDRiD 학습과 공식 test 최종 평가

이번 실험은 **IDRiD train에서 새로 학습하고 IDRiD test에서 최종 평가**한다.
EyePACS checkpoint를 그대로 적용하는 외부 검증과는 다른 실험이다.
내부 validation은 만들지 않으며, 공식 test는 gradient 갱신, scheduler,
epoch 선택 또는 checkpoint 선택에 사용하지 않는다.

## 높은 정확도를 읽기 전에 확인할 사항

- `metrics.json`의 `train.accuracy`, `final_train.accuracy`와 epoch의
  `train_accuracy`는 **학습 정확도**다. 413장 정도의 데이터에 장기 학습하면 매우
  높아질 수 있으며 이를 test 성능으로 보고하면 안 된다.
- 과거 `train_idrid_all_data.py`는 train 413장과 test 103장을 합친 516개 행을
  학습한 뒤 **같은 test 103장**을 재평가했다. `seen_test`의 높은 점수는
  resubstitution이며 독립 test 결과가 아니다.
- `train_idrid_fixed_iterations.py`는 원래부터 train-only였다. 높은 수치라는
  이유만으로 이 runner가 test로 학습했다고 결론 내릴 수는 없다. 원격 실행의
  `run_config.json`, `training_split.json`, `evaluation_metrics.json`으로 판별해야 한다.

이번 수정에서는 전체 데이터 진단 runner의 기본 실행을 차단했다.
`run_idrid_remote.sh`도 내부 validation을 사용하는 이전 runner 대신
**fixed-iteration train-only runner**를 실행한다. Epoch 로그는
`[TRAIN ONLY; TEST NOT EVALUATED]`, 최종 평가는 `[FINAL OFFICIAL TEST]`로 구분한다.
최종 평가가 끝나기 전에는 `evaluation_status`가 `pending_training`이다.

## 원격에서 실행할 명령어

CUDA/PyTorch가 설치된 학습 환경을 활성화한 뒤 원격 프로젝트 디렉터리에서 실행한다.
아래 명령은 기존 원본 데이터와 과거 실험 결과를 삭제하지 않는다.

```bash
git switch learnable_dist
git pull --ff-only origin learnable_dist
python -m pip install -r requirements-idrid.txt

# 이미 내려받은 data/idrid/raw를 다시 검사하고 prepared CSV를 갱신한다.
python scripts/prepare_idrid.py --data-dir data/idrid --overlap-policy report

# validation 없이 train-only 학습 -> 마지막 checkpoint -> test 103장 평가
python scripts/train_idrid_fixed_iterations.py \
  --data-dir data/idrid --device cuda \
  --output-dir checkpoints/idrid_heldout_audited_seed42_run01

# 학습 종료 후 저장 예측을 대조하여 test 결과만 출력한다.
python scripts/report_idrid_test.py checkpoints/idrid_heldout_audited_seed42_run01
```

출력 폴더가 이미 있으면 덮어쓰지 않으므로 새 실행에는 `run02`처럼 다른 이름을 쓴다.
기존 train+test 학습 checkpoint를 이어서 사용하지 않고 ImageNet 가중치에서 새로 시작한다.
이미 실행 중인 프로세스는 코드를 갱신해도 바뀌지 않는다. 새 코드는 새 프로세스로 실행한다.

데이터가 아직 없다면 Kaggle 인증을 원격 컴퓨터에 설정한 뒤 다음을 사용한다.

```bash
IDRID_PYTHON=python bash scripts/run_idrid_remote.sh \
  --device cuda --output-dir checkpoints/idrid_heldout_audited_seed42_run01
```

다른 경로의 데이터를 사용할 때는 `--data-dir /path/to/idrid`를 준비/학습 양쪽에
동일하게 지정한다. 기존에 추출한 Kaggle 폴더가 별도 위치에 있다면 먼저 다음을 실행한다.

```bash
python scripts/prepare_idrid.py --source-dir /path/to/extracted-idrid \
  --data-dir data/idrid --overlap-policy report
```

## 최종 test의 실제 구성 확인

| 공식 분할 | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | 전체 |
|---|---:|---:|---:|---:|---:|---:|
| train | 134 | 20 | 136 | 74 | 49 | 413 |
| test | 34 | 5 | 32 | 19 | 13 | 103 |

데이터 준비는 `B. Disease Grading`의 `Retinopathy grade`만 읽는다.
다른 과제의 영상/황반부종 라벨을 섞지 않는다. Train/test에서 `IDRiD_001` 같은
파일명이 반복되므로 **공식 split 폴더와 파일명**으로 연결한다.

학습 시작 시 다음을 확인한다.

1. Prepared CSV의 checksum과 실제 영상 파일의 SHA-256을 다시 계산한다.
   준비 이후 파일을 덮어썼다면 학습을 중단하고 재준비를 요구한다.
2. 각 경로가 실제 공식 Training/Testing 폴더에 속하는지 검사한다.
3. EXIF 방향을 적용한 RGB 픽셀의 checksum도 계산한다. 파일 메타데이터나
   lossless 저장 형식만 달라 파일 SHA-256이 다른 동일 영상도 확인한다.
4. Test와 같은 파일 **또는 같은 픽셀**을 가진 사본은 **학습 쪽에서만 제외**한다.
   Test 103장과 정답은 보존한다. 예를 들어 사본 한 장이 있으면 실제 학습은
   412장, test는 그대로 103장이다. 원본 파일/라벨은 삭제하거나 고치지 않는다.
5. 최종 평가 직전 실제 학습 멤버와 test의 ID·경로·파일·픽셀 해시가 겹치지 않는지
   검사한다. 평가 loader가 요청된 test 영상과 정답을 정확히 한 번씩 내놓았는지
   검증한 뒤에만 결과를 저장한다.

`--overlap-policy report`는 원본 분할을 준비 단계에서 보존하는 설정이다.
현재 train-only runner는 그 사본을 실제 학습 풀에서 제외하므로,
report 옵션이 test를 학습에 포함하도록 허용하는 것은 아니다.

정확한 픽셀 일치는 JPEG 재압축 등으로 픽셀이 달라진 유사 영상이나 동일 환자의
다른 촬영을 모두 탐지하지 않는다. 공개 CSV에 환자 ID가 없어 환자 단위 독립성까지
보장할 수 없다. 과거 test를 이미 학습하거나 반복 확인한 이력도 새 실행으로 지워지지 않는다.

## 모델과 학습 예산

기존 train-only 설정의 모델과 학습 예산을 유지한다.

- EfficientNetV2-S, ImageNet 초기화, GeM, Spatial Attention 미사용, 입력 224×224.
- PCOL/SCOLw 0.1/0.1, RMSE/MMNP 1/2, 회귀 입력은 projection concat.
- Seed 42, batch 24, **22 updates/epoch × 500 epochs = 11,000 updates**.
  층화 미니배치는 train 안에서 반복 표집하며 test로 배치를 채우지 않는다.
- 초기 간격 U[0.5, 1.0), margin lr=1e-3. Phase 1 최소 50/최대 100 epochs,
  변화량 1e-5 이하가 10회 유지되면 간격을 고정한다.
- Validation/early stopping 없음. Scheduler는 **train RMSE**만 사용한다.
- 마지막 `final.pt`를 다시 읽고 test를 평가한다. Test 점수로 checkpoint를 고르지 않는다.

기본 설정은 `configs/hybrid_idrid_fixed_iterations_no_spatial.yaml`이다.
이전 결과와 섞이지 않도록 기본 출력 폴더에는 `_audited_seed42`가 포함된다.

## Balanced Softmax 모델을 평가할 경우

현재 브랜치의 BS 모델도 같은 train-only 검사를 사용한다.
해당 모델을 돌리던 실험이라면 위 학습 명령 **대신** 다음을 실행한다.

```bash
python scripts/train_idrid_balanced_softmax.py \
  --data-dir data/idrid --device cuda \
  --output-dir checkpoints/idrid_bs_heldout_audited_seed42_run01
python scripts/report_idrid_test.py checkpoints/idrid_bs_heldout_audited_seed42_run01
```

`configs/hybrid_idrid_balanced_softmax.yaml`은 원본 브랜치
`gem_no_spatial_0814_bs_ablation`의 `03fd7e0`에 따른 classification head와
`CE(logits + log(class_counts), y)` 보조 손실을 사용한다.
PCOL/SCOLw는 0.5/0.1, BS 계수는 0.1이다. 일반 설정과 PCOL 계수도 다르므로
이 둘의 차이를 BS 하나의 효과로 해석하지 않는다. `--classification-weight 0.9`는
명시한 별도 실험에서만 사용한다. BS 클래스 개수는 사본 제외 후 실제 train에서만 계산한다.
최종 등급은 원래 구현처럼 **회귀 출력을 반올림/0–4 제한**하여 얻는다.
Classification head의 argmax로 평가 방식을 바꾸지 않는다.

## 어떤 파일을 논문 결과로 사용해야 하는가

- `evaluation_metrics.json`의 **`test`**: 최종 official test Accuracy, Macro,
  QWK, MAE, 클래스별 정확도와 혼동행렬. `evaluation_status: complete`,
  `official_test_used_for_training: false`, `evaluation_is_independent: true`,
  `evaluation_samples: 103`을 확인한다.
- `predictions.csv`: test 103장 각각의 `sample_id`, `split`, 정답, 예측 등급,
  연속 예측값, `overlaps_train`. 모든 행이 `test:*`/`official_test`여야 한다.
- `per_class_metrics.csv`: 각 정답 클래스의 정답 수/표본 수와 정확도(recall), MAE.
- `test_metrics.csv`: test 전체의 Accuracy, Macro Accuracy, QWK와 MAE 한 행.
- `confusion_matrix.csv`: test 전체 5×5 혼동행렬.
- `training_split.json`: 실제 train/test 멤버, 제외한 학습 사본, 픽셀 해시와 클래스 개수.
- `run_config.json`: 실제 설정과 코드 commit, 원본 데이터 정보, 영상 재검사 결과.
- `metrics_history.json`: 학습 경과. 여기에 있는 epoch 정확도를 test로 사용하지 않는다.

`report_idrid_test.py`는 test 멤버십/103장 분포를 검사하고, 저장 예측에서
Accuracy·Macro·QWK·MAE·혼동행렬을 다시 계산해 summary와 일치하는지 확인한다.
Train-only 로그만 있거나 `seen_test`인 예전 결과는 독립 test 점수로 출력하지 않는다.

## Test 전체 성능과 정답 표본만의 t-SNE

학습이 끝난 뒤 `final.pt`로 test를 평가할 때 위 지표와 함께 다음을 자동 저장한다.
성능 지표는 **test 전체 103장**에서 계산한다. 클래스별 CSV에는 기존 정확도/MAE와
함께 precision, recall, F1도 기록한다(정의할 수 없는 값은 빈 칸/null).

| 파일 | 내용 |
|---|---|
| `tsne_test_correct_only.png` | 정답을 맞힌 test 표본만 사용한 600-dpi 그림 |
| `tsne_test_correct_only.pdf` | 같은 그림의 벡터 PDF |
| `tsne_test_correct_only.csv` | 포함한 test ID, 정답/예측, 2차원 좌표 |
| `tsne_test_correct_only_embeddings.npz` | 정답 표본만의 고차원 특징과 ID |
| `tsne_test_correct_only_metadata.json` | 전체/정답 개수, 클래스별 포함 개수, seed/PCA/perplexity |

회귀 출력에 기존과 동일한 반올림과 0–4 제한을 적용해 정답 여부를 결정한다.
PCOL/SCOL 출력을 각각 L2 정규화한 뒤 연결한 특징을 사용하며,
**오답을 제외한 후에 PCA와 t-SNE를 적합**한다. Seed 42를 사용한다.
정답이 없는 클래스도 범례와 metadata에 n=0으로 표시한다.
정답이 3개 미만이거나 선택된 특징이 모두 같으면 좌표를 만들어내지 않고
건너뛴 사유와 안내 그림을 저장한다.
이 그림은 정답 부분집합의 정성적 시각화이며 전체 test의 클래스 분리도를 뜻하지 않는다.

### 이미 학습이 끝났거나 이전 코드로 학습 중인 경우

기존 학습을 다시 시작하거나 prepare할 필요가 없다. 현재 학습이 `final.pt`를
저장한 후 다음 명령으로 **평가만** 실행한다. 저장된 모델 설정과 학습/test 목록을
읽고, checkpoint를 그대로 재로딩한다. ImageNet 가중치도 다시 받지 않는다.

```bash
git pull --ff-only origin learnable_dist
python scripts/evaluate_idrid_test.py \
  --run-dir checkpoints/idrid_heldout_audited_seed42_run01 \
  --device cuda
python scripts/report_idrid_test.py \
  checkpoints/idrid_heldout_audited_seed42_run01/test_analysis
```

`--run-dir`에는 실제 기존 결과 폴더를 지정한다. 기본 출력은 그 폴더 안의
`test_analysis/`이며 원래 checkpoint/지표를 덮어쓰지 않는다.
이미 분석한 폴더가 있으면 `--output-dir checkpoints/idrid_test_analysis_run02`처럼
새 경로를 지정한다. BS 모델도 checkpoint의 설정으로 자동 구성한다.
Train+test로 학습한 과거 `all_data` checkpoint는 독립 test로 바꾸어 평가하지 않는다.

## 과거 진단 runner

`train_idrid_all_data.py`는 과거 결과 해석을 위해 보존하지만 기본 실행은 중단된다.
의도적인 학습 데이터 재평가 진단만 `--allow-test-training`을 명시할 수 있다.
그 결과는 여전히 `seen_test`이며 일반화 성능 보고에 사용하지 않는다.
이전 내부 validation runner는 `train_idrid.py`, 장기 설정은
`train_idrid_random_margin.py`이다. 현재 요청의 실행은 위 fixed-iteration runner다.

## 로컬 동작 검사

```bash
python -m pytest tests/test_idrid.py tests/test_idrid_fixed_iterations.py \
  tests/test_idrid_balanced_softmax.py tests/test_idrid_evaluation_integrity.py -q
python scripts/smoke_test_idrid_fixed_iterations.py
python scripts/smoke_test_idrid_balanced_softmax.py
```

Smoke test는 합성 영상/작은 head/사전학습 미사용으로 실제 EfficientNetV2-S의
역전파, 간격 고정, 저장·재로딩과 평가를 확인한다. 실제 원격 IDRiD/CUDA 성능을 대신하지 않는다.
