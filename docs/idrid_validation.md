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

논문은 ResNet-50 + 분류 손실 + MMNP를 사용한다. 여기서는 **우리 Hybrid 모델**을
유지하므로 CLOC 자체의 완전한 재현은 아니다. 간격 수렴에 따른 단계 전환,
모델 선택 절차, ImageNet 정규화 및 증강의 세부 강도도 구현 선택이다.
논문에서 명시하지 않은 회전 범위/증강 확률을 논문의 설정으로 단정하지 않는다.

`configs/hybrid_idrid_efficientnet_v2_s.yaml`에 모든 조건을 기록했다.
회전 ±30°, 수평/수직 반전 각 확률 0.5, blur 확률 0.5(커널 5, σ 0.1–2.0),
color jitter 확률 0.5(brightness/contrast/saturation 0.2, hue 0.05)를 기본으로 한다.
평가에는 resize, tensor 변환, ImageNet 정규화만 적용한다.
EyePACS의 검은 테두리 crop은 이 프로토콜에 추가하지 않았다.

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

## 학습과 평가 절차

1. 공식 학습 413장만 등급별 층화하여 내부 학습 330장/검증 83장으로 나눈다.
   공개 라벨에는 환자 ID가 없으므로 환자 단위 독립 분할이라고 주장하지 않는다.
2. 내부 검증 Accuracy가 가장 높은 Phase 2의 epoch를 선택한다. 동률에서는
   discrete-grade MAE가 작은 쪽, 다시 동률이면 먼저 나온 epoch를 선택한다.
   Scheduler는 내부 검증 MAE를 사용하고, Phase 2에서 10회 연속 개선이 없으면 종료한다.
   전체 epoch 상한은 500이며, Phase 1은 기존 모델의 간격 수렴 조건으로 20–30 epoch에 종료한다.
3. 동일한 seed로 모델을 새로 초기화하고 공식 학습 **413장 전체**로 재학습한다.
   위에서 선택한 epoch 수, 단계 전환 시점, epoch별 학습률을 그대로 재생한다.
   간격 값 자체는 전체 학습 세트에서 새로 학습하고 Phase 2에 고정한다.
4. 저장된 모델을 다시 읽은 뒤 공식 테스트 **103장**을 한 번 평가한다.
   테스트 Accuracy/MAE는 epoch, scheduler, checkpoint 선택에 사용하지 않는다.

이 내부 선택 후 전체 재학습 절차는 테스트 세트를 학습 선택에서 분리하기 위해
추가한 프로토콜이다. 논문에 동일한 내부 검증 절차가 명시되어 있다는 의미는 아니다.
최종 `best.pt`는 이 절차로 정한 학습 기간의 마지막 모델이며 테스트 성능으로 선택하지 않는다.
기존 결과가 있는 출력 디렉터리에는 덮어쓰지 않으므로 새 실험에는 다른 `--output-dir`을 쓴다.

## 저장 결과

기본 결과 디렉터리: `checkpoints/hybrid_idrid_efficientnet_v2_s/`

- `evaluation_metrics.json`: 최종 테스트 Accuracy, MAE, Class 0–4 정확도, 혼동행렬 등.
- `predictions.csv`: 각 영상의 정답, 회귀 출력, 예측 등급, 절대오차.
- `best.pt`: 공식 학습 413장으로 재학습한 최종 모델.
- `run_config.json`: 설정, 데이터 출처/개수, 실행 Git commit 및 환경.
- `selection_split.json`, `selection_history.json`, `training_plan.json`: 내부 선택 근거.
- `metrics_history.json`: 전체 학습 세트 재학습의 epoch별 손실과 간격.
- `selection_best.pt`: 내부 선택 단계의 모델. 최종 테스트 대상은 `best.pt`이다.

Accuracy는 클래스별 정확도의 단순 평균이 아니라 전체 정답 수/N이다.
MAE는 회귀 출력을 반올림하고 0–4 범위로 제한한 **예측 등급과 정답 등급의 절대차 평균**이다.
반올림은 기존 프로젝트와 같은 PyTorch/NumPy의 nearest-even 규칙을 사용한다.
`continuous_mae`는 참고용으로 별도 저장하며 논문의 등급 MAE와 혼동하지 않는다.
`accuracy`는 0–1 비율, `accuracy_percent`는 0–100 퍼센트이다.

현재 로컬 구현에 있던 CLOC 식 (2)의 MMNP 합산 수정도 원격에 포함한다.
`mmnp_reduction: mean`은 각 anchor의 positive–negative 쌍 손실을 **합산한 뒤
anchor 간 평균**을 구한다. 과거 쌍 전체를 평균하던 실행과는 손실 크기가 다르므로
기존 실행의 loss 값이나 가중치 효과를 직접 같은 척도로 비교하지 않는다.

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
