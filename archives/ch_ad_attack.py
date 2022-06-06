import numpy as np
import torch

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

device = 'cuda:2' if torch.cuda.is_available() else 'cuda:3'

def untargeted_fgsm(model, img, eps):
    """
    untargeted FGSM의 적대적 예제 생성 함수

    :model: 학습된 인공지능 모델.
            공격자는 인공지능 모델의 모든 파라미터 값을 알고있음.
    :img:   적대적 예제로 바꾸고자 하는 이미지 데이터
    :eps:   적대적 예제에 포함될 noise 크기 결정.
            eps가 크면 클 수록, 적대적 공격은 성공률이 높지만,
            적대적 예제의 시각적 표현이 높아지는 단점이 있음.
    :return: tensor 형태의 적대적 예제

    """

    fgsm_data = fast_gradient_method(model, img, eps, np.inf)

    return fgsm_data[0]

def targeted_cw(model, img, target):

    cw_data = carlini_wagner_l2(model, img, 10, y=torch.tensor([target]).to(device), targeted=True)

    return cw_data

def untargeted_pgd(model, img, eps):

    pgd_data = projected_gradient_descent(model, img, eps)

    return pgd_data