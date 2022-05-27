# ad_masking
적대적 예시 방어 (XAI 활용 + Model activation 수정)

사용법: python main.py --params params.yaml
* mnist, cifar10 변경 -> params.yaml 수정

현재, mnist, cifar-10 모델 학습

ToDoList:
  - pgd attack
  - captum 사용
  -

### Prerequisites
Run the following:
```
$ pip3 install -r requirements.txt
```

### Training models
Run the following:
```
$ python3 train.py --params params.yaml
```