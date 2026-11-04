# EyePACS Class 3 한 장의 Spatial Attention 맵 추출

Spatial Attention 적용 실험의 `best.pt`로 추론을 한 번 수행합니다. 체크포인트에 저장된 모델 구조와 입력 전처리를 사용하며, 기본적으로 같은 분할 설정의 **검증 세트에서 정답이 Class 3인 영상 한 장**을 이미지 ID 순서대로 선택합니다. 예측이 틀려도 선택된 영상을 그대로 출력합니다.

저장소 루트에서 실행합니다.

```bash
python scripts/export_spatial_attention.py \
  --run-dir checkpoints/hybrid_eyepacs_efficientnet_v2_s_e1_e2_gem384 \
  --class-id 3 \
  --split val \
  --device cuda \
  --output-dir outputs/spatial_attention_class3
```

체크포인트 위치가 다르면 `--run-dir` 대신 `--checkpoint /경로/best.pt`를 사용합니다. `--run-dir`은 그 폴더의 `best.pt`를 선택합니다. 기존 no_spatial 또는 GeM-only 체크포인트는 공간 주의 가중치가 없으므로 사용할 수 없습니다.

`384_gem_spatial_cloc110-2`는 폴더명에 `spatial`이 있지만 실제 저장 설정은 `spatial_attention: false`입니다. 폴더명 대신 설정과 실제 가중치를 확인해야 합니다. 다음 명령은 네트워크 추론 없이 현재·옆 프로젝트의 실제 체크포인트를 검사하고, 사용 가능한 `READY` 항목의 절대 경로와 실행 명령을 출력합니다.

```bash
python scripts/export_spatial_attention.py --list-runs
```

다른 위치에 체크포인트가 있으면 `--search-root /실제/경로/checkpoints`를 추가합니다. `BEST MISSING`은 해당 폴더에 `best.pt`가 없음을, `NO SPATIAL WEIGHTS`는 실제 공간 주의 가중치가 없음을 의미합니다. 없는 가중치를 새로 초기화하여 학습된 Attention 맵으로 출력하지 않습니다.

상대 경로의 폴더가 현재 checkout에 없으면 같은 상위 디렉터리의 다른 프로젝트에서 **이름이 정확히 같은** `checkpoints/실험명`을 찾습니다. 예를 들어 `DrGrading-idrid`에서 실행하더라도 옆 `DrGrading/checkpoints/실험명`을 찾을 수 있습니다. 후보가 여러 개면 절대 경로를 지정해야 합니다. 저장된 상대 데이터 경로는 체크포인트가 속한 프로젝트를 기준으로 해석합니다.

저장된 데이터 경로를 현재 컴퓨터에 맞게 지정하려면 다음 옵션을 추가합니다.

```bash
  --labels-csv /경로/trainLabels.csv \
  --image-root /경로/train
```

다른 Class 3 영상을 확인할 때는 `--sample-index 1`을 사용하거나, CSV의 이미지 ID를 `--image-id 123_left`로 지정합니다. `--split test`는 저장된 설정에 외부 테스트 fold가 있는 경우 사용할 수 있습니다. 학습에 사용한 YAML을 따로 지정해야 하는 이전 체크포인트에는 `--config /경로/training.yaml`을 추가합니다. CPU에서 실행할 때는 `--device cpu`를 사용합니다.

출력 폴더에는 다음 파일을 저장합니다.

- `comparison.png`, `comparison.pdf`: 입력, 실제 공간 가중치, 입력 위의 중첩 결과를 나란히 배치한 그림
- `overlay_original.png`: 원본 해상도의 안저 영상 위에 중첩한 결과. 추론에서 잘라낸 바깥 영역은 원본을 유지
- `overlay_input.png`: 실제 모델 입력의 크기와 좌표에 맞춘 중첩 결과
- `spatial_attention.npy`: 특징 맵 해상도에서 추출한 원래 sigmoid 공간 가중치
- `spatial_attention_input.npy`: 모델 입력 크기로 bilinear 보간한 공간 가중치
- `spatial_attention.png`, `input.png`: 색상 맵과 전처리된 입력
- `metadata.json`: 영상 ID, 정답·예측 등급, 연속 예측값, 체크포인트 SHA256, 분할 설정, 좌표와 색상 범위

맵은 모델의 `spatial_attention.conv` 출력을 sigmoid에 통과시켜 얻은 실제 공간 가중치입니다. 그림의 색상 범례는 이 가중치의 실제 최솟값과 최댓값을 표시하며, 원래 수치는 `.npy`에 보존합니다. 여러 영상을 비교할 때는 `metadata.json`의 색상 범위와 원래 가중치를 함께 확인합니다.

논문에서는 입력 영상의 병변, 시신경유두, 주요 혈관, 망막 경계와 조명 영역에 대한 반응을 비교해 서술합니다. 이미지 한 장의 맵은 해당 사례의 집중 영역을 보여주며, 여러 영상에서 반복되는 위치 편향은 같은 절차로 얻은 추가 사례를 함께 비교하여 분석합니다.
