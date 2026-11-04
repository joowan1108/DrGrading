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
