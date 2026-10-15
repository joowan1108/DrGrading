# IDRiD 추가 데이터 실험

현재 Hybrid 모델(EfficientNetV2-S, PCOL/SCOLw/RMSE, 학습 가능한 간격,
Two-phase Learning, Spatial Attention, GeM)을 IDRiD에서 새로 학습한다.
EyePACS 체크포인트를 그대로 평가하는 zero-shot 외부 검증과는 다른 실험이다.

## 논문과 일치시키는 조건

[CLOC 논문 §4.1–4.2 및 부록 A.4](https://arxiv.org/pdf/2504.17813)의
공식 IDRiD 학습 413장/테스트 103장 분할, 224×224 입력, 수평·수직 반전,
회전, Gaussian blur, IDRiD용 color jitter를 적용한다. 등급별 개수는 다음과 같다.

| 분할 | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | 전체 |
|---|---:|---:|---:|---:|---:|---:|
| 학습 | 134 | 20 | 136 | 74 | 49 | 413 |
| 테스트 | 34 | 5 | 32 | 19 | 13 | 103 |

모델과 학습은 기존 프로젝트의 **Hybrid 모델 및 `train_hybrid_ordinal.py` 방식**을
기준으로 한다. IDRiD 분할과 요청한 영상 크기/증강 종류를 적용하며,
CLOC의 ResNet-50, 500 epochs, 학습 정확도 95% 기반 단계 전환은 사용하지 않는다.
논문에서 명시하지 않은 회전 범위/증강 확률을 논문의 설정으로 단정하지 않는다.

`configs/hybrid_idrid_efficientnet_v2_s.yaml`에 모든 조건을 기록했다.
회전 ±30°, 수평/수직 반전 각 확률 0.5, blur 확률 0.5(커널 5, σ 0.1–2.0),
color jitter 확률 0.5(brightness/contrast/saturation 0.2, hue 0.05)를 기본으로 한다.
평가에는 resize, tensor 변환, ImageNet 정규화만 적용한다.
EyePACS의 검은 테두리 crop은 이 프로토콜에 추가하지 않았다.

PCOL과 SCOLw 가중치는 이전 실험에 맞춰 각각 `alpha: 0.1`, `beta: 0.1`을 사용한다.
총손실은 `0.1 * PCOL + 0.1 * SCOLw + 1.0 * RMSE + 2.0 * MMNP`이다.
실행 시작 시 가중치를 출력하며, 매 epoch가 끝나면 각 손실의 원래 값과 가중치를
곱한 기여도를 함께 출력한다. 실제 실행 값은 `run_config.json`에도 저장한다.

각 epoch 시작 시 `[Phase 1: joint]` 또는 `[Phase 2: frozen]`, epoch 번호,
네트워크 학습률과 margin 학습 여부를 출력한다. 진행 막대에도 phase를 표시한다.
`[Phase 1 -> Phase 2]` 전환 로그에는 전환 사유, 고정 간격, 다음 phase의 시작
epoch 및 학습률/scheduler/checkpoint 선택 상태 초기화 여부를 표시한다.
`metrics_history.json`의 각 epoch에도 `phase`와 margin 고정 여부를 저장한다.

기존 코드의 기준 commit은 IDRiD 추가 전 `e677bfb`이다. 이 설정에서 PCOL/SCOLw
계수만 사용자 지정 `0.1/0.1`로 적용했다. Adam(lr=0.001, weight_decay=0.0001),
최대 75 epochs, validation RMSE 기반 ReduceLROnPlateau(factor=0.2, patience=5),
조기 종료 patience=13, temperature=1.0, ImageNet 정규화, GeM/Spatial Attention,
dropout=0.2와 projection 연결 입력을 사용한다. 역빈도 SCOLw 표본 가중치,
L2 embedding 정규화, 대조 손실의 평균 1 margin 정규화와 stop-gradient도 유지한다.
초기 인접 간격은 기존 지정값 `[0.385, 0.451, 0.253, 0.330]`, 간격 학습률은 1e-5다.
IDRiD 전용 기준은 PCOL/SCOLw 계수와 데이터/분할 어댑터이므로 기존 EyePACS 실행
설정 파일은 변경하지 않는다.

## Random margin · 장기 학습 재실험

새 실행 파일은 `scripts/train_idrid_random_margin.py`, 설정은
`configs/hybrid_idrid_random_margin_long.yaml`이다. 기존 75-epoch 실험 설정은
비교를 위해 보존한다. 원격 프로젝트 폴더에서 기존 준비 데이터를 사용한다.

```bash
python scripts/train_idrid_random_margin.py --device cuda
```

| 조건 | 기존 IDRiD 실험 | Random margin 재실험 |
|---|---|---|
| 인접 간격 초기화 | 고정된 4개 값 | 각 간격을 U[0.5, 1.0)에서 추출 |
| 간격 학습률 | 1e-5 | 1e-3 |
| 전체 epoch 상한 | 75 | 500 |
| Phase 1 최소/최대 | 20/30 | 50/100 |
| 간격 변화량 허용값 | 1e-3 | 1e-5 |
| 연속 안정 epoch 수 | 3 | 10 |
| Phase 2 조기 종료 patience | 13 | 50 |
| LR scheduler patience | 5 | 10 |

U[0.5, 1.0) 초기화와 Softplus 양수 제약은 [CLOC §3.3](https://arxiv.org/pdf/2504.17813)
설명을 따른다. 학습 중 간격을 0.5 이상으로 고정하는 제약은 아니며 0보다 큰 값으로
계속 학습한다. seed=42로 전체 초기화가 재현되므로 같은 설정/seed로 다시 실행하면
같은 초기 간격이 나온다. 학습 기간, margin 학습률, 수렴 허용값과 patience는 이번
Hybrid 재실험을 위한 선택이며 논문의 전체 학습 프로토콜을 재현한다는 의미는 아니다.

PCOL/SCOLw 가중치 0.1/0.1, RMSE/MMNP 가중치 1/2, MMNP 쌍 평균, temperature,
ImageNet 정규화, 배치 구성, 데이터 분할, 영상 증강과 모델 구조는 기존 IDRiD 설정과 같다.
Phase 1은 최소 50 epochs 이전에는 종료되지 않으며, 이후 최대 인접 간격 변화량이
1e-5 이하로 10회 연속 유지되거나 100 epochs에 도달하면 고정한다.
Phase 2는 검증 RMSE가 50회 연속 개선되지 않으면 조기 종료하며, 전체 500 epochs를
반드시 수행한다는 의미는 아니다. checkpoint와 최종 평가는 이전처럼 검증 RMSE만으로
선택한 `best.pt`를 사용하고, 테스트 지표로 학습 설정을 자동 선택하지 않는다.

기본 결과 폴더는 `checkpoints/hybrid_idrid_random_margin_long_a01_b01_seed42/`이다.
`initial_margins.json`에 실제 초기값과 초기화 방식/학습률을 저장한다.
매 epoch 로그와 `metrics_history.json`에는 현재 간격, 초기값 대비 변화량,
epoch 간 최대 변화량과 안정 epoch 수를 저장한다. 기존 결과를 덮어쓰지 않는다.

```bash
python scripts/smoke_test_idrid.py --config configs/hybrid_idrid_random_margin_long.yaml
```

위 smoke test는 random 설정을 읽되 합성 데이터와 두 epoch로 단축해 실제 모델의
초기화·역전파·간격 고정·checkpoint 평가를 확인한다. 500-epoch 성능 검증이 아니다.

## 원격 컴퓨터에서 실행

원격 GPU 컴퓨터의 프로젝트 디렉터리에서 실행한다. 이 명령은 명령을 실행한
컴퓨터에 데이터를 받는다. 로컬에서 SSH로 원격 다운로드를 실행하는 스크립트가 아니다.

```bash
git switch learnable_dist
git pull --ff-only origin learnable_dist
python -m pip install -r requirements-idrid.txt
```

CUDA에 맞는 PyTorch가 설치된 학습 환경을 활성화한다. Kaggle 인증은 원격 컴퓨터에
설정한다. [공식 Kaggle 인증 안내](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md)를
따라 API 토큰 또는 기존 `kaggle.json`을 사용하며, 인증 파일은 저장소에 넣지 않는다.

Linux/WSL에서는 다음 한 줄로 다운로드, 준비, 학습, 평가를 진행한다.

```bash
IDRID_PYTHON=python bash scripts/run_idrid_remote.sh --device cuda
```

데이터/결과 저장 위치를 바꾸려면 다음과 같이 실행한다.

```bash
IDRID_PYTHON=python IDRID_DATA_DIR=/mnt/data/idrid \
  bash scripts/run_idrid_remote.sh --device cuda \
  --output-dir checkpoints/idrid_run_01
```

Windows PowerShell 또는 단계별 실행:

```text
python scripts/prepare_idrid.py --download --data-dir data/idrid
python scripts/train_idrid.py --config configs/hybrid_idrid_efficientnet_v2_s.yaml --data-dir data/idrid --device cuda
```

이미 압축을 해제했다면 다운로드 대신 다음을 실행한다.

```text
python scripts/prepare_idrid.py --source-dir /path/to/extracted/kaggle-dataset --data-dir data/idrid
```

Kaggle ZIP은 원격의 `data/idrid/archive/`에 보관하고, `B. Disease Grading`만
`raw/`에 추출한다. Segmentation/Localization 영상과 라벨은 학습에 섞지 않는다.
준비 시 `Image name`과 `Retinopathy grade` 열을 읽으며 황반부종 라벨은 사용하지 않는다.
학습/테스트에 같은 파일명이 있으므로 **분할 폴더 + 파일명**으로 연결한다.
공식 클래스 분포, 누락/중복 영상, 두 분할 사이의 동일 파일 내용(SHA-256)을 검사한다.

### 동일한 이미지 내용이 학습/테스트에 포함된 경우

`Identical image contents`는 같은 파일명이 아니라 **파일 바이트의 SHA-256 일치**를
뜻한다. 기본 동작은 준비를 중단하고 `prepared/duplicate_audit.json`에 양쪽 파일
경로, 등급, 해시와 라벨 충돌 여부를 기록하는 것이다. 이 단계에서 중단되면
`manifest.json`이 없으므로 학습을 먼저 실행할 수 없다.

공급된 413/103 분할을 보존하며 중복을 명시한 결과를 얻으려면 다음처럼 실행한다.
기존 원격 `raw/`를 다시 검사하므로 데이터를 다시 다운로드하지 않는다.

```bash
python scripts/prepare_idrid.py --data-dir data/idrid --overlap-policy report && \
python scripts/train_idrid.py --data-dir data/idrid --device cuda
```

통합 스크립트에서는 `IDRID_OVERLAP_POLICY=report` 환경변수를 지정할 수 있다.
report 모드는 영상이나 정답 라벨을 삭제/수정하지 않는다. 전체 테스트 결과에는
학습 영상과의 중복이 포함되므로 독립적인 일반화 성능이라고 해석하지 않는다.
`evaluation_metrics.json`의 `data_integrity`에 중복 개수를 표시하고,
`test_without_train_overlap.metrics`에 해당 영상을 제외한 Accuracy/MAE도 저장한다.
후자는 표본 수가 103장보다 적어질 수 있어 논문의 원래 테스트 결과와 동일한 평가가 아니다.
모든 테스트 영상이 중복이면 해당 부분 지표는 `null`과 사유를 기록한다.
이 해시 검사는 바이트가 완전히 동일한 파일만 찾으며 재압축 영상이나 동일 환자를
모두 탐지하는 검사는 아니다. 클래스 개수 일치만으로 Kaggle 미러의 내용이 원본과
동일하다고 보장하지 않는다. 중복의 출처는 보고서를 바탕으로 별도 확인해야 한다.

별도 `git worktree --detach`로 실행 중인 폴더는 다음 명령으로 업데이트한다.
로컬 코드 변경이 있으면 Git이 중단하므로 해당 변경을 먼저 보존해야 한다.

```bash
git fetch origin
git merge --ff-only origin/learnable_dist
```

## 학습과 평가 절차

1. 공식 학습 413장만 등급별 층화하여 내부 학습 330장/검증 83장으로 나눈다.
   공개 라벨에는 환자 ID가 없으므로 환자 단위 독립 분할이라고 주장하지 않는다.
   학습 세트 내부에도 같은 내용의 파일이 있으면 SHA-256 그룹 단위로 층화 분할하여
   동일 영상이 내부 학습과 검증에 동시에 들어가지 않게 한다. 이 경우 정확한 분할
   개수는 달라질 수 있으며 `training_split.json`에 실제 목록과 방식을 기록한다.
   동일 내용에 서로 다른 등급이 붙어 있으면 그룹을 유지하는 후보 분할 중 모든
   등급이 포함되고 목표 크기에 가까운 분할을 고른다. 정답 라벨은 변경하지 않는다.
2. 같은 모델을 최대 75 epochs 학습한다. Phase 1에서는 네트워크와 간격을 함께
   학습하며 최소 20 epochs 이후 간격 변화량이 0.001 이하로 3회 유지되거나,
   30 epochs에 도달하면 간격을 고정하고 Phase 2로 전환한다.
3. Phase 2 전환 시 네트워크 학습률, scheduler, 최적 checkpoint 추적 상태와
   조기 종료 카운터를 초기화한다. **검증 RMSE 최솟값**으로 `best.pt`를 선택하고,
   scheduler도 검증 RMSE를 사용한다. 13회 연속 개선이 없으면 조기 종료한다.
   간격이 고정된 Phase 2에서도 경계 손실은 네트워크를 계속 최적화한다.
4. **선택한 `best.pt`를 다시 읽어** 공식 테스트 **103장**을 한 번 평가한다.
   테스트 Accuracy/MAE는 epoch, scheduler, checkpoint 선택에 사용하지 않는다.

최초 IDRiD 스크립트에 추가했던 학습 기간 선택 후 413장 전체 재학습(refit)은 제거했다.
최종 checkpoint는 내부 학습 세트(기본 330장)에서 학습한 모델이다. 즉 전체 413장을
모두 gradient 갱신에 사용한 실험으로 표기하지 않는다. 기본 설정의 epoch당 배치 수는
14개이며, 테스트 데이터는 학습이나 checkpoint 선택에 섞지 않는다.
기존 결과가 있는 출력 디렉터리에는 덮어쓰지 않으므로 새 실험에는 다른 `--output-dir`을 쓴다.

## 저장 결과

기본 결과 디렉터리: `checkpoints/hybrid_idrid_efficientnet_v2_s_existing_a01_b01/`

이전 IDRiD 학습 방식/가중치 실행 결과와 구분하기 위해 기본 경로를 변경했다.
이미 실행 중인 프로세스는 시작할 때 읽은 설정으로 계속 학습하므로,
새 가중치를 적용하려면 기존 실행을 종료하고 새 프로세스로 시작한다.
기존 체크포인트를 자동으로 이어서 학습하지 않는다.

- `evaluation_metrics.json`: 전체 테스트 및 학습 중복 제외 부분집합의 지표, 중복 정보.
- `predictions.csv`: 각 영상의 정답, 회귀 출력, 예측 등급, 절대오차, `overlaps_train` 표시.
- `best.pt`: 내부 검증 RMSE가 가장 낮은 Phase 2 모델.
- `last.pt`, `phase1_last.pt`: 마지막 epoch 및 단계 전환 직전 checkpoint.
- `best_metrics.json`, `metrics.json`: 최적 검증 결과와 최종 테스트 결과.
- `run_config.json`: 설정, 데이터 출처/개수, 실행 Git commit 및 환경.
- `training_split.json`: 내부 학습/검증 영상 목록 및 분할 방식.
- `metrics_history.json`: epoch별 학습/검증 지표와 간격, phase, best 여부.

Accuracy는 클래스별 정확도의 단순 평균이 아니라 전체 정답 수/N이다.
MAE는 회귀 출력을 반올림하고 0–4 범위로 제한한 **예측 등급과 정답 등급의 절대차 평균**이다.
반올림은 기존 프로젝트와 같은 PyTorch/NumPy의 nearest-even 규칙을 사용한다.
`continuous_mae`는 참고용으로 별도 저장하며 논문의 등급 MAE와 혼동하지 않는다.
`accuracy`는 0–1 비율, `accuracy_percent`는 0–100 퍼센트이다.

IDRiD에서는 `mmnp_pair_reduction: mean`, `mmnp_reduction: mean`으로 이전 실험의
경계 손실 계산을 사용한다. 각 anchor의 positive–negative 쌍 손실을 **평균**한 뒤
유효한 anchor 간 평균을 구한다. 쌍 수가 서로 다를 수 있으므로 모든 쌍을 한꺼번에
평균하는 것과는 다르다. 공통 loss 클래스의 기본값 `pair_reduction: sum`은 유지하여
이미 sum 방식으로 설정한 다른 실험이 자동으로 바뀌지 않게 했다.
이전 IDRiD의 쌍 합산 실행과는 손실 스케일이 다르므로 결과/설정을 구분한다.

## 데이터 다운로드 없는 동작 검사

```text
python -m pip install pytest
python -m pytest tests/test_idrid.py tests/test_learnable_margins.py -q
python scripts/smoke_test_idrid.py
```

Smoke test는 임시 디렉터리에 만든 합성 이미지 30장으로 실제 EfficientNetV2-S,
224×224 증강, GeM/Spatial Attention, 역전파, 간격 고정, checkpoint 재로딩,
Accuracy/MAE 저장까지 검사한다. CPU, batch 10, 작은 projection/regression head,
사전학습 미사용, 각 단계 1 epoch를 사용하므로 실제 장기 학습/원격 CUDA 검증을 대체하지 않는다.
데이터와 ImageNet 가중치를 다운로드하지 않으며 모든 결과에 synthetic 표식을 기록한다.
실제 IDRiD 성능은 원격 학습 완료 후에만 확인할 수 있다.
