# adversarial attack modules
import foolbox as fb
from torch.utils.data import DataLoader

import numpy as np
import torch

# helper function to obtain attack parameters
def atk_param(epsilon, targeted=False):
    param = {
        'FGSM'  : {'eps': epsilon, 'norm': np.inf,
                   'clip_min':0.0, 'clip_max':1.0, 'targeted': targeted},
        'PGD'   : {'eps': epsilon, 'eps_iter': 0.01, 'nb_iter': 40,
                   'norm': np.inf, 'clip_min':0.0, 'clip_max':1.0,
                   'sanity_checks': False, 'targeted': targeted},
        'CW'    : {'clip_min': 0.0, 'clip_max': 1.0, 'initial_const': 1e-2,
                   'n_classes': 10, 'binary_search_steps': 5, 'max_iterations': 1000,
                   'targeted': targeted},
        'SPSA'  : {'eps': epsilon, 'nb_iter': 40,
                   'clip_min':0.0, 'clip_max':1.0, 'targeted': targeted}
    }

    return param

# generate adversarial examples
def attack(dataset, model, bat_size, atk_method, epsilon, target=None, dir=None):
    """
    Generates adversarial example for the given model.

    :param dataset: attack dataset
    :param model: target model
    :param bat_size: batch size
    :param atk_method: method of adversarial attack
    :param epsilon: adversarial attack parameter
    :param target: target label (single for now)
    :param dir: directory to save and load adversarial examples
    """

    try:
        # try to load dataset
        adv_xs, adv_ys = torch.load(dir)

    except:
        # dataloader
        data_loader = DataLoader(dataset, bat_size)

        # generate attack parameters
        atk_fn = eval(atk_method)
        if target is not None:
            atk_p = atk_param(epsilon, targeted=True)[atk_method]
            t = torch.tensor(target)
            t_ = t.repeat(bat_size).cuda()
        else:
            atk_p = atk_param(epsilon, targeted=False)[atk_method]
            t_ = None

        adv_xs, adv_ys = [], []
        for (x,y) in data_loader:
            if target is not None and x.size(0) != bat_size:
                t_ = t.repeat(x.size(0)).cuda()
            adv_x = atk_fn(model, x.cuda(), **atk_p, y=t_)
            adv_y = model(adv_x)
            _, adv_y = torch.max(adv_y.data, 1)
            adv_xs.append(adv_x.cpu())
            adv_ys.append(adv_y.cpu())

        # concatenate
        adv_xs = torch.cat(adv_xs, 0)
        adv_ys = torch.cat(adv_ys, 0)

        # save results
        if dir is not None:
            torch.save((adv_xs, adv_ys), dir)

    return adv_xs, adv_ys