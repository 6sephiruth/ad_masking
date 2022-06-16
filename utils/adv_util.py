import torch
import foolbox as fb
import numpy as np

def atk_params(atk_method):
    pass


def attack(model, data_loader, method, epsilon, preprocessing, device):
    # TODO: for multiple epsilons?
    assert type(epsilon) == float

    # define fb torch model
    fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing)

    # get string variable for function name
    atk_fn = f"fb.attacks.{method}"
    atk = eval(atk_fn)()

    adv_xs = []
    atk_succ = []
    for (x,y) in data_loader:
        _, advs, success = atk(fmodel, x.to(device), y.to(device), epsilons=epsilon)
        adv_xs.append(advs.cpu())
        atk_succ.extend(success.cpu())

    robust_acc = 1 - np.mean(atk_succ)

    print(f"robust accuracy with attack method: {method}")
    print(f"  norm ≤ {epsilon:<6}: {robust_acc.item() * 100:4.1f} %")

    return torch.cat(adv_xs, 0), np.where(atk_succ)[0]