# 병변 마스크의 빈 박스와 Spatial Attention 비교

`export_spatial_lesion_boxes.py`는 병변 마스크의 각 연결 영역을 감싸는 **속이 빈 청록색 사각형**을 원본 영상과 Spatial Attention 중첩 영상에 그립니다. 마스크를 색으로 채우거나 병변 종류의 설명을 영상 위에 쓰지 않습니다. 작은 병변을 포함해 기본적으로 1픽셀 이상의 모든 연결 영역을 표시합니다.

EyePACS에는 기본 병변 마스크가 없으므로 기본 실행은 **IDRiD Segmentation의 실제 마스크가 있는 Class 3 영상**을 선택하고, 지정한 EyePACS CBAM 체크포인트로 추론합니다. 모델은 그 체크포인트의 300×300 입력·전처리·Average Pooling을 사용합니다. IDRiD 학습을 새로 수행할 필요는 없습니다.

## 그림만 추출

`DrGrading-idrid` 프로젝트 루트에서 실행합니다.

```bash
git pull --ff-only origin learnable_dist
python scripts/export_spatial_lesion_boxes.py \
  --run-dir ../DrGrading/checkpoints/hybrid_eyepacs_efficientnet_v2_s_softplus_squared_mmnp_spatial_cbam \
  --idrid-dir data/idrid \
  --class-id 3 --segmentation-split test --sample-index 0 \
  --device cuda \
  --output-dir outputs/spatial_lesion_boxes_class3
```

다른 사례는 `--sample-index 1`, `2` 등으로 지정합니다. 기본 선택은 경로 순서이며, Attention이 병변을 잘 찾거나 못 찾는 정도로 사례를 자동 선별하지 않습니다. 마스크가 있는 test 부분집합에서 Class 3을 찾지 못하면 `--segmentation-split all`로 범위를 넓힐 수 있습니다. 이 split은 **Segmentation의 split**으로, Grading train/test와 동일한 분할을 의미하지 않습니다.

## 기존 데이터 재사용

기존 `data/idrid/prepared/`의 Grading CSV와 원본 영상, 기존 Segmentation 원본·마스크를 읽습니다. Task A와 Task B의 숫자 ID를 직접 연결하지 않고, **EXIF 방향을 반영한 실제 RGB 픽셀 해시가 같은 경우에만** DR 등급을 연결합니다. 연결된 Grading sample ID는 메타데이터에 기록합니다.

기존 준비 코드는 ZIP에서 Task B만 추출하므로 `A. Segmentation`이 아직 없을 수 있습니다. 이때 이미 내려받은 `data/idrid/archive/*.zip`에 Task A가 있으면 **그 ZIP의 Segmentation 부분만 추출**합니다. Grading CSV를 다시 준비하거나 데이터를 다운로드하지 않습니다.

Segmentation을 다른 곳에 풀어 두었다면 다음 옵션을 추가합니다.

```bash
  --segmentation-root /실제/경로/A.%20Segmentation
```

기존 폴더와 ZIP 양쪽에 Segmentation이 없으면 원본 마스크가 필요하다는 오류를 출력합니다. 마스크를 Attention에서 역으로 만들지 않습니다.

## 출력

- **`lesion_box_comparison.pdf` / `.png`**: 원본+병변 박스, Spatial Attention 중첩, Attention+병변 박스의 3패널 그림. 폭 7.2인치, PNG 600 dpi, PDF의 박스는 벡터로 저장
- **`lesion_boxes_original.png`**: 원본 해상도의 영상에 빈 박스만 표시
- **`attention_lesion_boxes_original.png`**: 원본 해상도의 Attention 중첩 영상에 같은 빈 박스 표시
- `lesion_boxes.csv`: 병변 종류, 연결 영역의 픽셀 수, 정확한 박스 좌표와 표시용 박스 좌표
- `lesion_box_metadata.json`: 영상·체크포인트·마스크의 경로와 해시, 등급 연결 방법, 예측, 전처리와 표시 설정
- `spatial_attention.npy`: 모델에서 추출한 원래 sigmoid 공간 가중치

박스 좌표는 EXIF 방향을 반영한 **원본 해상도** 기준이고, `x1`, `y1`은 영역에 포함하지 않는 끝 좌표입니다. 기본 `--box-padding 6`은 빈 박스 테두리가 작은 병변을 가리지 않도록 사방 6픽셀 여백을 더합니다. 정확한 경계 상자는 `x0…y1`, 표시용 경계 상자는 `draw_x0…draw_y1`에 각각 저장합니다. 마스크 자체와 Attention 값은 변경하지 않습니다.

마스크는 원본 크기와 일치해야 하며, 크기가 다르면 자동 resize하지 않고 오류를 냅니다. 원본의 EXIF 방향 보정은 마스크에도 동일하게 적용합니다. Attention은 모델이 실제 사용한 crop 영역을 원본 좌표로 되돌려 표시하고, crop 밖 영역은 원본 영상으로 유지합니다.

`--box-padding`, `--line-width`로 박스 표시를 조정할 수 있습니다. `--min-mask-area`의 기본값은 1이며, 값을 높이면 그 픽셀 수보다 작은 연결 영역을 제외합니다. 연결 영역은 병변 종류별 8방향 연결로 계산합니다. IDRiD의 MA, HE, EX, SE 마스크를 사용하고 시신경유두 마스크는 병변 박스로 사용하지 않습니다.

## 직접 지정한 영상과 마스크

같은 원본 영상에 대응하는 마스크가 따로 있을 때는 다음과 같이 지정할 수 있습니다. 이 모드는 IDRiD 준비 파일을 읽지 않습니다.

```bash
python scripts/export_spatial_lesion_boxes.py \
  --run-dir ../DrGrading/checkpoints/hybrid_eyepacs_efficientnet_v2_s_softplus_squared_mmnp_spatial_cbam \
  --image /경로/original.jpg \
  --mask /경로/original_HE.tif \
  --mask /경로/original_MA.tif \
  --true-class 3 --device cuda \
  --output-dir outputs/spatial_custom_lesion_boxes
```

`--true-class`는 아는 경우에만 지정하며, 생략하면 정답 등급은 미지정으로 저장합니다. 파일 이름이 `*_MA`, `*_HE`, `*_EX`, `*_SE`이면 CSV에 그 병변 종류를 기록하고, 그 외 이름은 `lesion`으로 기록합니다.

## 그림 추출 후 IDRiD 학습까지 이어서 실행

기존 결합 명령에 **`--lesion-boxes`**를 추가합니다.

```bash
python scripts/run_spatial_and_idrid_best.py \
  --lesion-boxes \
  --spatial-run ../DrGrading/checkpoints/hybrid_eyepacs_efficientnet_v2_s_softplus_squared_mmnp_spatial_cbam \
  --source-run checkpoints/hybrid_eyepacs_efficientnet_v2_s_384_gem_no_spatial \
  --data-dir data/idrid --device cuda \
  --output-dir outputs/spatial_and_idrid_best_seed42 \
  --skip-completed
```

그림은 `spatial_class3/lesion_box_comparison.pdf`에 저장합니다. 그림 추출 완료 후 IDRiD의 train+test 학습과 train-only 학습·평가·정답 표본 t-SNE 추출을 실행합니다. IDRiD 학습에는 앞서 선택한 **384 입력·GeM·no-spatial·Balanced Softmax 미적용 설정**을 사용합니다. `--skip-completed`는 같은 설정으로 평가까지 완료된 IDRiD 실험만 재사용합니다.
