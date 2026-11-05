# GAP·GeM 비교 그림 추출

풀링만 달리한 두 **학습 완료 모델**을 같은 원본 영상에 적용합니다. 재학습이나 데이터 다운로드는 수행하지 않습니다.

기본 체크포인트는 기존 통제 실험의 다음 두 폴더에 있는 best.pt입니다. 현재 프로젝트나 같은 상위 디렉터리의 다른 프로젝트에서 **정확히 같은 폴더명**을 찾습니다.

| 모델 | 실험 폴더 | 기존 검증 정확도 |
|---|---|---:|
| GAP | hybrid_eyepacs_efficientnet_v2_s_384_avgpool_no_spatial | 79.59% |
| GeM | hybrid_eyepacs_efficientnet_v2_s_384_gem_no_spatial | 81.45% |

실제 학습 가중치를 엄격하게 불러오고, 두 모델의 전처리·구조·분할·학습 설정을 검사합니다. Spatial Attention 및 Balanced Softmax가 포함된 모델은 이 비교에 사용하지 않습니다. 표의 정확도는 저장된 과거 검증 결과이며, 이 스크립트가 평가 세트 전체에서 새로 계산한 정확도가 아닙니다.

## EyePACS의 모든 클래스(Class 0–4) 그림 추출

DrGrading-idrid 프로젝트 루트에서 실행합니다. 결과 폴더 등 프로젝트 내부의 하위 디렉터리에 있다면 첫 번째 명령으로 루트로 이동합니다.

~~~bash
cd "$(git rev-parse --show-toplevel)"
git pull --ff-only origin learnable_dist

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python scripts/compare_gap_gem_maps.py \
  --classes 0 1 2 3 4 --samples-per-class 1 --split val \
  --device cpu
~~~

다른 터미널에서 GPU 학습을 진행 중이면 위 CPU 명령을 사용할 수 있습니다. GPU로 추출할 때는 --device cuda로 변경합니다.

--classes를 생략해도 기본값은 모든 클래스 0–4입니다. 위 명령은 클래스마다 한 장씩 총 다섯 장을 비교합니다. 저장된 seed·환자 단위 분할·외부 fold를 재구성하고, 각 정답 클래스에서 영상 ID 순서로 사례를 선택합니다. 두 모델의 예측이나 맵을 보고 성공 사례만 선별하지 않습니다. --sample-index 1로 다음 사례를, --samples-per-class 3으로 클래스마다 세 장씩 총 열다섯 장을 추출할 수 있습니다. 특정 클래스만 보려는 경우에만 --classes 2 3처럼 지정합니다.

python: can't open file '.../outputs/.../scripts/compare_gap_gem_maps.py' 오류는 현재 디렉터리가 프로젝트 루트가 아니라 결과 폴더라는 뜻입니다. 위 cd 명령을 먼저 실행하거나, 스크립트의 절대 경로를 지정합니다. 이 오류는 체크포인트 문제가 아니며 새 학습은 필요하지 않습니다.

결과는 outputs/gap_gem_maps_날짜_시각/에 자동 저장합니다. --output-dir을 생략하면 매번 새 폴더를 만들므로 기존 결과와 충돌하지 않습니다.

EyePACS에는 기본 병변 위치 마스크가 없으므로 이 모드에서는 원본과 두 모델의 맵을 표시하며, 병변 상자·병변 영역 내 비율을 임의로 생성하지 않습니다.

## 앞서 만든 Spatial Attention 그림과 같은 영상·병변 상자 사용

기존 추출 결과의 lesion_box_metadata.json에 원본 영상과 마스크 경로가 들어 있습니다. 합성된 class2_spatial.png 등을 입력하지 않고 이 메타데이터로 **실제 원본 영상과 실제 마스크**를 재사용합니다.

파일 위치는 다음 명령으로 확인합니다.

~~~bash
rg --files outputs -g 'lesion_box_metadata.json'
~~~

그중 비교하려는 사례의 경로를 지정합니다. 예를 들어 기존 Class 3 결과가 아래 위치에 있다면:

~~~bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python scripts/compare_gap_gem_maps.py \
  --case-metadata outputs/spatial_lesion_boxes_class3/lesion_box_metadata.json \
  --device cpu
~~~

--case-metadata 뒤에 Class 2·3의 JSON 경로를 함께 적으면 두 사례를 한 PDF에 배치합니다. 원본 픽셀과 마스크의 해시를 확인하고, 메타데이터에 기록된 정답 등급·자료명을 그대로 유지합니다. 특히 EyePACS로 학습한 모델을 IDRiD Segmentation 영상에 적용했다면 영상의 자료명은 IDRiD로 유지됩니다.

## 원본 영상과 마스크 직접 지정

~~~bash
python scripts/compare_gap_gem_maps.py \
  --image /실제/경로/original.jpg \
  --true-class 3 \
  --mask /실제/경로/original_HE.tif \
  --mask /실제/경로/original_MA.tif \
  --device cuda
~~~

--mask와 --true-class는 선택 사항입니다. 마스크는 원본 영상과 크기가 일치해야 하며, EXIF 방향과 실제 추론 crop을 반영합니다. 상자는 속이 빈 청록색이고 기본 여백은 원본 기준 6픽셀입니다. 수치 분석에는 상자 내부가 아니라 원래 병변 마스크의 픽셀을 사용합니다.

## 체크포인트·데이터 위치가 다를 때

~~~bash
python scripts/compare_gap_gem_maps.py \
  --gap-checkpoint /실제/경로/384_avgpool_no_spatial/best.pt \
  --gem-checkpoint /실제/경로/384_gem_no_spatial/best.pt \
  --labels-csv /실제/경로/eyepacs/trainLabels.csv \
  --image-root /실제/경로/eyepacs/train \
  --classes 0 1 2 3 4 --device cuda
~~~

저장된 모델 후보와 검증 정확도만 확인하려면:

~~~bash
python scripts/compare_gap_gem_maps.py --list-runs
python scripts/compare_gap_gem_maps.py --list-runs --search-root ../DrGrading/checkpoints
~~~

저장된 설정의 차이가 있으면 실행 전에 항목별로 알려줍니다. --allow-config-differences를 명시하면 탐색 목적으로 추출할 수 있지만, 그림 제목과 메타데이터에 설정 차이를 표시합니다. 입력 크기나 정규화가 다른 두 모델은 이 옵션으로도 통제 비교하지 않습니다.

## 출력과 해석

- **gap_gem_gradcam.pdf**, **gap_gem_gradcam_page01.png**: 원본 / GAP / GeM의 회귀 출력 Grad-CAM.
- **gap_gem_gradient_activation.pdf**, **gap_gem_gradient_activation_page01.png**: 같은 배치의 gradient × activation 비교. 공간별 기울기를 유지하므로 비선형 GeM의 위치별 민감도도 확인할 수 있습니다.
- **gap_gem_pooling_moments.pdf**, **gap_gem_pooling_moments_page01.png**: **한 개의 동일한 GeM 특징 맵**에서 p=1과 학습된 p의 상대적 집계 기여를 비교합니다. GAP 모델의 예측 맵과 구분되는 동작 원리 그림입니다.
- **comparison_metrics.csv**: 사례별·모델별 예측값과 등급, 정답 여부, 병변 영역 내 기여도 비율·면적 보정 비율.
- **comparison_metadata.json**: 체크포인트 경로·해시·epoch·기존 검증 결과, 실제 적용한 설정, 사례 선택 방식과 그림의 해석.
- **case_*/gap_maps.npz**, **case_*/gem_maps.npz**: 원래 특징 맵 해상도의 수치. 부호 있는 기여도 맵도 보관합니다.

기본적으로 두 회귀 기여도 방법을 모두 추출합니다. --method gradcam 또는 --method gradient_activation으로 한 종류만 선택할 수 있습니다. 각 PDF에는 한 페이지당 최대 세 사례를 배치합니다. PNG는 600 dpi이고, PDF의 글자와 병변 상자는 벡터입니다.

우리 모델은 회귀 모델이므로 **반올림·범위 제한 전의 연속 중증도 출력**을 미분합니다. 양의 기여도는 중증도를 높이는 방향의 민감도를 뜻하며, 특정 클래스의 확률이나 학습된 Spatial Attention 가중치를 뜻하지 않습니다. 표기용 예측 등급만 기존 평가 방식과 동일하게 반올림하고 0–4로 제한합니다.

기여도 맵은 양수 부분을 각 맵의 최댓값으로 나눈 동일한 규칙으로 표시합니다. 따라서 같은 색의 절대 기울기가 모델 사이에서 같다는 뜻은 아닙니다. 양의 기여도가 전혀 없으면 원본 영상으로 표시하고 수치에 그 상태를 기록합니다.

병변 면적 보정 비율은 다음과 같습니다.

~~~
병변 기여도 비율 = 병변 마스크 안의 양의 기여도 합 / 추론 crop 전체의 양의 기여도 합
면적 보정 비율 = 병변 기여도 비율 / crop에서 병변 마스크가 차지하는 면적 비율
~~~

마스크가 없거나, 해당 맵의 양의 기여도 합이 0이면 정의할 수 없는 수치는 비워 둡니다. 집계 원리 그림은 채널별 x^p / sum(x^p)를 평균한 분포이며, 분류·회귀 예측에 대한 기여도와는 별도입니다. 학습된 p가 1보다 작은 경우도 그대로 표시합니다.

그림에서 병변 위치와의 대응, 경계·배경의 반응, 예측 변화가 어떻게 나타나는지 확인한 뒤 실제 결과에 맞춰 원고를 작성합니다. 이 저장소에는 해당 두 모델의 학습 가중치가 포함되지 않으므로 실제 비교 그림은 가중치가 있는 원격 환경에서 추출합니다.

방법 참고: [Grad-CAM](https://arxiv.org/abs/1610.02391), [GeM Pooling](https://arxiv.org/abs/1711.02512).
