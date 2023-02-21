# Vehicle Re-identification using FAST-REID
- 서로 다른 두 영상 사이에 등장하는 동일 차량을 구분해내는 문제의 Sub task

## Task

- 공개된 사전 학습된 모델 중에 어떤 것이 프로젝트에 가장 적합한 지 평가
  - Fast-ReID의 Vehicle Re-Identification을 위한 Pretrained model 중 어떤 것이 Main task를 해결하기에 가장 적합한지 평가
  - Pretrained Model은 각각 아래의 데이터 셋으로 학습
    - [VehicleID](https://www.v7labs.com/open-datasets/vehicleid)
    - [VeRi-776](https://www.v7labs.com/open-datasets/veri-dataset)
    - [VERI-WILD](https://www.v7labs.com/open-datasets/veri-wild)

## dataset

- [한국 이미지 (차량)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=80)

![그림 13](https://user-images.githubusercontent.com/18918072/220297641-4b8dd69e-774a-434b-bea4-2f3371637c2e.png)

  - 빛 조건, 다양한 시점, 해상도 등의 다양성을 제공
  - 100종의 차량과 5만 장의 이미지로 구성됨
  - Why?
    - 외국의 차량 데이터로 학습된 pretrained model들이 국내 차량에 대해서도 얼마나 유효한 모델을 확인하기 위해서 사용
    - 학습된 dataset과 유사한 다양성을 제공하는 국내 차량 이미지를 사용하여 Pretrained model을 평가함
    
## Data Preprocessing

1. Data Selection

- 차량 전체를 포함한 이미지만 선택

![그림 16](https://user-images.githubusercontent.com/18918072/220298485-4dee4917-de71-447d-a5d7-a82fe46b831e.png)

- why?
  - CCTV 영상에서는 대부분의 차량이 차량 전체 모습을 비추고 있기 때문에 차량 전체에 대한 이미지가 필요

2. Image Cropping

- Annotation 정보를 통해 이미지에서 차량 부분만 Cropping

![그림 17](https://user-images.githubusercontent.com/18918072/220299036-638c8726-1d1d-41bf-929d-97e674cdad7c.png)

3. Data Grouping

- Target data의 시나리오는 CCTV에서 차가 주행중인 상황으로 차량의 방향이 일반적으로 잘 바뀌지 않음
- 따라서 차량의 각 측면에 대해 분리하여 검증하는 것이 필요함


![그림 19](https://user-images.githubusercontent.com/18918072/220301302-8f8c973d-eddf-490c-9d00-b1367862dfcb.png)

![그림 20](https://user-images.githubusercontent.com/18918072/220301329-a3cf5726-3ac1-447f-b3e8-3e56e36e355f.png)

![그림 21](https://user-images.githubusercontent.com/18918072/220301336-68fb433a-d6b5-4ed7-8f9a-420ea281778f.png)

## Model Evaluation

- 카테고리 샘플링을 통한 Rank-k Evaluation
  - 각 카테고리의 첫 번째 이미지를 Query Image로 지정
  - 각 카테고리의 나머지 이미지를 Gallery Image로 지정
  - 각 카테고리마다 쿼리를 수행함
    - 현재 그룹에서는 Query를 가져오고, 각 카테고리의 gallery로부터 이미지를 한 장씩 샘플링
    - 가져온 이미지들을 Re-ID 모델을 통해 feature를 추출
    - Query feature와 각 카테고리의 gallery feature와의 cosine similarity 계산
    - gallery feature를 유사도 기준으로 정렬
    - 현재 그룹에서 가져온 gallery 이미지가 유사도 순위에 따라 rank1, rank5, rank10을 계산
  - 모든 카테고리의 쿼리로 얻은 rank1, rank5, rank10 값을 통해 성능을 평가
  
## Result
![그림 29-1](https://user-images.githubusercontent.com/18918072/220308280-12cba9fa-a17e-4ebf-a9b1-b767d5c9be89.png)

- 모든 시점에서 VERI-WILD가 높은 성능을 나타냄
- Main Task에서 관심 시나리오인 정면에 대해서는 VeRi와 VehicleID가 유사한 성능을 보였음
    
