# Structured-Unstructured-Pruning

![image](https://github.com/user-attachments/assets/cac54be3-d1d5-4ff4-971a-7dc309f3d0b9)

기본 모델 비교 실험 Task로 Image Classification을 진행하였다. Structure pruning은 L1 Norm으로 중요도를 계산하여 각 Convolution의 channel을 제거하는 channel pruning을 진행하였다. Unstructured pruning은 개별 가중치의 절대값을 기준으로 가중치를 제거하는 Magnitude-based Pruning을 진행하였다. 실험에서는 CIFAR10 데이터를 사용하여 비교실험을 진행하였다. 기본 Base 모델로는 ResNet18을 사용하였다. 해당 실험에서 pretrain 모델을 가져와 Structured/Unstructured Pruning ratio 50%를 진행하였다. 결과물을 보면 Structured pruning이 속도와 모델 성능 모두 향상된 걸 확인할 수 있다. 