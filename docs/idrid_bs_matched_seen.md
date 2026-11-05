# 기존 82.52% IDRiD 실험과 조건을 맞춘 Balanced Softmax

`scripts/train_idrid_seen_balanced_softmax.py`는 기존 `hybrid_idrid_all_data_gem_no_spatial_seed42/run_config.json`에서 실제 설정을 읽어 BS 보조 분류 head와 손실만 추가합니다. 기존 `train_idrid_balanced_softmax.py`의 train-only·alpha 0.5 설정과 다른 실행입니다.

보통 기존 실행에서 다음 값을 그대로 복사합니다.

| 항목 | 값 |
|---|---|
| 학습 데이터 | 공식 train 413 + test 103 = 516행, test 복사본 제외 없음 |
| 평가 데이터 | 같은 공식 test 103장 |
| 모델·입력 | EfficientNetV2-S, 224×224, GeM, no spatial |
| Seed·학습 | seed 42, batch 24, 500 epoch |
| PCOL / SCOL / RMSE / MMNP | 0.1 / 0.1 / 1 / 2 |
| MMNP 집계·margin lr | pair-mean / 0.001 |
| Phase 1 | 최소 50, 최대 100 epoch; 원래 수렴 기준 유지 |
| BS 추가 손실 가중치 | 기본 0.1, `--classification-weight`로 변경 가능 |
| 최종 예측 | 원래 회귀 출력의 반올림·0–4 제한; BS head는 보조 손실 |

새 모델은 원래 설정에 따라 ImageNet 초기화에서 학습합니다. 원래 학습된 비-BS 체크포인트 가중치를 옮기지 않습니다. 원래 `initial_margins.json` 또는 평가 기록의 초기 margin 값을 재사용해 BS head 추가로 인한 초기 간격 차이를 줄입니다. 초기값 파일이 없으면 같은 값을 확인할 수 있는 원래 실행 경로를 지정해야 합니다. BS의 클래스 수는 516행으로 구성한 실제 학습 풀에서 계산합니다.

## 학습량을 줄이고 주기적으로 평가하기

원래 500 epoch는 비-BS 실행의 학습 예산을 복사한 값입니다. 기본 22배치/epoch에서는 11,000회 배치 업데이트에 해당합니다. 짧은 예산의 별도 실험은 `--epochs`로 지정합니다. 예를 들어 200 epoch는 4,400회 업데이트입니다. 원래 Phase 1 최대 기간이 100 epoch이므로 총 epoch는 100보다 커야 합니다.

```bash
python scripts/train_idrid_seen_balanced_softmax.py \
  --source-run checkpoints/hybrid_idrid_all_data_gem_no_spatial_seed42 \
  --data-dir data/idrid --device cuda \
  --classification-weight 0.1 \
  --epochs 200 --test-every-epochs 100 \
  --tsne-sweep \
  --output-dir "checkpoints/idrid_bs_200ep_$(date +%Y%m%d_%H%M%S)"
```

`--test-every-epochs 100`은 100·200 epoch에서 평가합니다. **미니배치 iteration 100회마다** 평가하려면 대신 `--test-every-steps 100`을 사용합니다. 기본 22배치/epoch에서는 약 4.55 epoch 간격이며, 200 epoch 실행에서 총 44회 주기 평가합니다. 두 간격 옵션은 함께 사용하지 않습니다.

평가 대상은 학습에 포함된 공식 test 103장 전체입니다. 각 평가에서 전체·매크로 정확도, QWK, MAE와 Class 0–4 정확도를 로그와 `periodic_test_metrics.csv` / `.json`에 저장합니다. 시점별 예측과 클래스별 CSV는 `periodic_test/step_00000100/`처럼 배치 iteration으로 이름 붙인 폴더에 저장합니다. 이 시점의 가중치는 메모리에서 평가하며, `final.pt`를 평가한 것으로 기록하지 않습니다. 주기 평가에서는 t-SNE나 별도 대형 체크포인트를 만들지 않습니다. t-SNE는 전체 학습 종료 후 생성합니다.

주기 평가값은 scheduler·early stopping·checkpoint 선택에 사용하지 않습니다. 평가 전후에 모델 학습 모드와 Python/NumPy/Torch RNG를 보존합니다. 같은 seed의 합성 실행에서 주기 평가를 켰을 때와 껐을 때 최종 모델·간격 가중치가 동일한지 검사합니다.

`run_config.json`에 실제 기간과 간격을, provenance의 `epoch_budget`에 원래 기간과 변경 여부를 기록합니다. BS만 200 epoch로 줄인 결과는 기존 500 epoch 비-BS 결과와 학습량도 달라집니다. 같은 짧은 예산으로 BS 효과를 비교하려면 비-BS도 같은 기간으로 실행합니다.

```bash
python scripts/train_idrid_all_data.py --allow-test-training \
  --data-dir data/idrid --device cuda \
  --epochs 200 --test-every-epochs 100 \
  --output-dir "checkpoints/idrid_nonbs_200ep_$(date +%Y%m%d_%H%M%S)"
```

기존 500 epoch 실행은 그대로 보존하며, 위 명령은 새 출력 폴더에서 새 모델을 학습합니다.

## 이미 완료한 실행의 Phase 1 체크포인트 평가

이전 실행에 `phase1_last.pt`가 있다면 재학습 없이 그 시점의 성능을 확인할 수 있습니다. 아래 명령은 그 파일에 기록된 실제 epoch를 출력하고 test 전체 지표와 t-SNE를 새 폴더에 저장합니다. `phase1_last.pt`를 `final.pt`로 바꾸거나 원래 결과를 덮어쓰지 않습니다.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python scripts/idrid_seen_class1_tsne.py \
  --run-dir checkpoints/hybrid_idrid_all_data_gem_bs_matched_seed42 \
  --checkpoint-name phase1_last.pt \
  --data-dir data/idrid --device cpu \
  --output-dir "outputs/idrid_bs_phase1_$(date +%Y%m%d_%H%M%S)"
```

원래 실행에서 100·200·300·400 epoch의 가중치를 각각 저장하지 않았다면, 최종 가중치만으로 그 시점들의 test 점수를 소급 계산할 수는 없습니다.

WSL/Bash의 repo 루트에서 실행합니다.

```bash
git pull --ff-only origin learnable_dist
python scripts/train_idrid_seen_balanced_softmax.py \
  --source-run checkpoints/hybrid_idrid_all_data_gem_no_spatial_seed42 \
  --data-dir data/idrid --device cuda \
  --output-dir checkpoints/hybrid_idrid_all_data_gem_bs_matched_seed42 \
  --classification-weight 0.1 --tsne-sweep
```

순서는 516행 학습 → 마지막 `final.pt` 저장 → test 103장 평가 → 같은 체크포인트 재추론으로 QWK·MAE·클래스별 지표와 t-SNE 저장입니다. 내부 validation·held-out 실험은 만들지 않습니다. 기존 결과는 덮어쓰지 않고 새 출력 경로를 사용합니다.

출력은 다음과 같습니다.

- `final.pt`: BS 적용 모델 가중치
- `balanced_softmax.json`: 학습 풀의 클래스 수와 BS 설정
- `source_configuration_comparison.json`: 원래 비-BS 설정과 실제 BS 설정
- `training_split.json`: train과 test를 포함한 학습 풀 목록
- `analysis_class1_all/evaluation_metrics.json`: 재추론한 test 전체 지표
- `analysis_class1_all/tsne_class1_all_other_correct.png` / `.pdf`: EyePACS와 같은 기본 none scaling, perplexity 30, 1000 iteration, Barnes–Hut 보기
- `analysis_class1_all/sweep/tsne_sweep.png` / `.pdf`: 12개 설정 비교

Class 1은 전부, 나머지 클래스는 정답 표본만 t-SNE에 사용합니다. 색상은 정답 클래스이며, C1 오분류를 포함한 모든 점을 EyePACS와 같은 원형으로 표시합니다. 비-BS 모델에도 같은 scaling·perplexity·iteration을 사용해 비교합니다. BS에서 정답 표본 수가 달라지면 같은 선택 규칙으로 그림에 들어가는 표본 수도 달라집니다.

학습은 끝났고 그림만 다시 만들려면 다음처럼 실행합니다.

```bash
python scripts/idrid_seen_class1_tsne.py \
  --run-dir checkpoints/hybrid_idrid_all_data_gem_bs_matched_seed42 \
  --data-dir data/idrid --device cpu \
  --scaling none --perplexity 30 --max-iter 1000 --sweep \
  --output-dir outputs/idrid_bs_seen_analysis_run02
```
