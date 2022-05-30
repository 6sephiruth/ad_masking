# ad_masking
Neuron 영향도 기반 masking을 사용한 적대적 예시 방어 기법 (XAI 활용 + Model activation 수정)

### 요구 패키지 설치
Our implementation runs on Python 3.8+. In order to install required packages, run the following:
```
$ pip3 install -r requirements.txt
```

### 모델 학습
```
python train.py --params params.yaml
```
params.yaml을 통해 MNIST, CIFAR-10, SVHN 모델 학습 가능.

### TODOs
TODO lists moved to [here](./TODOs.md)