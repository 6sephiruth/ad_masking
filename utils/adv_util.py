import torch
import foolbox as fb
import numpy as np

def atk_params(atk_method):
    pass


def attack(data_loader, model, atk_method, atk_epsilon, preprocessing, device):
    # TODO: for multiple epsilons?
    assert type(atk_epsilon) == float

    # define fb torch model
    fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing)

    # get string variable for function name
    atk_fn = f"fb.attacks.{atk_method}"
    atk = eval(atk_fn)()

    adv_xs = []
    atk_succ = []
    for (x,y) in data_loader:
        _, advs, success = atk(fmodel, x.to(device), y.to(device), epsilons=atk_epsilon)
        adv_xs.append(advs.cpu())
        atk_succ.extend(success.cpu())

    #print(atk_succ)
    robust_acc = 1 - np.mean(atk_succ)

    print(f"robust accuracy with attack method: {atk_method}")
    print(f"  norm ≤ {atk_epsilon:<6}: {robust_acc.item() * 100:4.1f} %")
    #print(f"robust accuracy: {robust_acc}")

    return torch.cat(adv_xs, 0)