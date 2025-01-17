# Structured-Unstructured-Pruning

## Project Overview
이 프로젝트는 딥러닝 모델 경량화 방법인 Structured Pruning과 Unstructured Pruning의 성능을 비교 분석하는 실험을 수행합니다. CIFAR10 데이터셋에서 ResNet18을 기본 모델로 사용하여 두 가지 pruning 방식의 효과를 비교합니다.

### Pruning Method
- **Structured Pruning**: L1 Norm 기반의 Channel Pruning
  - Convolution 레이어의 채널 단위로 pruning을 수행
  - 채널의 중요도를 L1 Norm으로 계산하여 하위 50% 제거
  
- **Unstructured Pruning**: Magnitude-based Pruning
  - 개별 가중치 단위의 pruning 수행
  - 가중치의 절대값을 기준으로 하위 50% 제거

## Key Features
- ResNet18 모델을 사용한 CIFAR10 이미지 분류
- Structured/Unstructured Pruning 구현 및 적용
- 모델 성능 평가 (정확도, 파라미터 수, 추론 시간)
- Pruning 전후 성능 비교 분석

## Getting Started

### Prerequisites
```bash
torch >= 1.7.0
torchvision >= 0.8.0
numpy
```

### Installation
1. 저장소 클론
```bash
git clone https://github.com/username/structured-unstructured-pruning.git
cd structured-unstructured-pruning
```

2. 필요 패키지 설치
```bash
pip install torch torchvision numpy
```

### Usage
```bash
python main.py
```

## Project Structure
```
├── config.py           # 설정 파일 (하이퍼파라미터, 실험 설정)
├── data_loader.py      # 데이터 로딩 및 전처리
├── main.py            # 메인 실행 파일
├── model_utils.py     # 모델 학습/평가 유틸리티
├── pruning_methods.py # Pruning 구현
└── README.md
```

## Configuration
config.py에서 다음 설정들을 조정할 수 있습니다:
```python
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.1
PRUNING_AMOUNT = 0.5    # Pruning 비율 (50%)
INFERENCE_RUNS = 100    # 추론 시간 측정 반복 횟수
```

## Experiment Process
1. ResNet18 Pretrained 모델 로드
2. CIFAR10 데이터셋에서 초기 학습 수행
3. Structured/Unstructured Pruning 적용 (50%)
4. Pruning된 모델 재학습
5. 성능 평가 및 비교
   - 모델 정확도
   - 파라미터 수 및 스파시티
   - 추론 시간

## Results
<div align="center">
  <img src="https://github.com/user-attachments/assets/cac54be3-d1d5-4ff4-971a-7dc309f3d0b9" alt="Structured-Unstructured-Pruning">
</div>
실험 결과, Structured Pruning이 모델의 속도와 성능 모두에서 개선된 결과를 보여주었습니다. 구체적인 결과는 모델을 실행하면 확인할 수 있으며, 다음 정보들이 출력됩니다:  
- 모델별 정확도  

- 파라미터 수와 Sparsity
  
- 추론 시간 및 속도 향상률  
