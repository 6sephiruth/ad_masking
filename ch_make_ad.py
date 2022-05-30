
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
from torchvision import datasets, transforms


from ch_ad_attack import *
from utils import *

from tqdm import trange

import pickle

record_cw = np.zeros(10, dtype='int')

if exists(f'./dataset/targeted_cw_data'):
    targeted_cw_data = pickle.load(open(f'./dataset/targeted_cw_data','rb'))

else:

    print("targeted cw 데이터에 적합한 데이터 검색 중")

    for i in range(10):
        for batch_idx, (data, target) in enumerate(test_data):

            if target.numpy()[0] == i:
                
                for j in range(10):
                        
                    data = data.type(torch.cuda.FloatTensor).to(device)
                    
                    cw_result = targeted_cw(model, data, j)
                    pred = model(cw_result).cpu().data.numpy().argmax()

                    if pred != j:
                        break
                    record_cw[i] = batch_idx

            if record_cw[i] != 0:
                break



print("targeted attack 데이터 생성 중")

for i in range(10):
    for j in range(10):

        img = torch.unsqueeze(torch.tensor(origin_data[i]).to(device), 0)
        img = img.type(torch.cuda.FloatTensor)

        cw_data =  targeted_cw(model, img, j)
        cw_data = cw_data.cpu()
        
        targeted_cw_data[i][j] = cw_data[0]

        print("정상 label {}를 적대적 예제 {}로 변환. ".format(i, j))

    pickle.dump(targeted_cw_data, open(f'./dataset/targeted_cw_data','wb'))
