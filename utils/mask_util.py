# layer attributions
from captum.attr import (
    LayerActivation,
    LayerConductance,
    InternalInfluence,
    LayerGradientXActivation,
    LayerGradCam,
    LayerDeepLift,
    LayerDeepLiftShap,
    LayerGradientShap,
    LayerIntegratedGradients,
    LayerFeatureAblation,
    LayerLRP
)

from functools import reduce
from itertools import pairwise
from torch.utils.data import DataLoader
from torch.nn.utils import prune
import numpy as np
import torch

# TODO: filter and average only mislabeled examples
def get_attr(model, data_loader, attr_method, transform=None, get_y=False):
    attributions = {}

    # aggregate attribution for each layer
    for n,l in model.named_children():
        res = []
        attr = eval(attr_method)(model, l)
        for (x,y) in data_loader:
            # apply transform
            x = transform(x) if transform else x

            if attr_method == 'LayerActivation':
                res.append(attr.attribute(x.cuda()).detach().cpu())
            else:
                if get_y:
                    y = model(x.cuda())
                    _, y = y.max(1)
                res.append(attr.attribute(x.cuda(), target=y.cuda()).detach().cpu())

        print(n, res)

        res = torch.cat(res, 0)
        attributions[n] = res
        #print(n, res)
        #print(res.size())

    return attributions

def get_mask(norm_attr, adv_attr, k, exclude=[]):
    masks = {}
    for n in norm_attr.keys():
        # get attributions
        atr_n = norm_attr[n]
        atr_a = adv_attr[n]

        if n in exclude or k == 0:
            masks[n] = torch.ones(atr_n.size()[1:], dtype=bool)
        else:
            atr_diff = (atr_a > atr_n).type(torch.float)
            atr_diff = torch.mean(atr_diff, 0)
            #print(max(atr_diff), min(atr_diff))
            #print(diff_mean.size())
            #atr_a = attr_adv[n]
            #atr = atr_n - atr_a
            th0 = torch.quantile(atr_diff, k)
            print(th0)
            masks[n] = atr_diff >= th0
            print(masks[n])

    return masks


def adv_masks(attr_norm, attr_adv, k, exclude=[], mode='diff'):
    masks = {}
    if mode == 'top_k':
        top_k_norm = top_k(attr_norm, k, exclude)
        top_k_adv = top_k(attr_adv, k, exclude)

        for n,n_act in top_k_norm.items():
            if n in exclude:
                masks[n] = torch.ones_like(n_act)
            else:
                a_act = top_k_adv[n]
                masks[n] = torch.logical_or(n_act, ~a_act)

    elif mode == 'diff':
        for n,atr_n in attr_norm.items():
            if n in exclude or k == 0:
                masks[n] = torch.ones_like(atr_n, dtype=bool)
            else:
                atr_a = attr_adv[n]
                atr = atr_n - atr_a
                #atr = minmax(atr_a) - minmax(atr_n)
                th0 = torch.quantile(atr, k)
                masks[n] = atr >= th0

    elif mode == 'freq':
        for n,atr_n in attr_norm.items():
            if n in exclude or k == 0:
                masks[n] = torch.ones_like(atr_n, dtype=bool)
            else:
                atr_a = attr_adv[n]
                atr = atr_n - atr_a
                #atr = minmax(atr_a) - minmax(atr_n)
                th0 = torch.quantile(atr, k)
                masks[n] = atr >= th0
                #masks[n] = torch.logical_or(atr <= 0, atr <= th0)

    return masks

def minmax(a):
    mi, mx = torch.min(a), torch.max(a)
    if mi == mx:
        return a-mi

    return (a-mi)/(mx-mi)

def top_k(attr, k, exclude, mode='all_except'):
    if mode == 'all':
        attr_all = torch.cat([v.flatten().cpu() for n,v in attr.items()], 0)
        th0 = torch.quantile(attr_all, 1-k)

    elif mode == 'all_except':
        attr_all = torch.cat([v.flatten().cpu() for n,v in attr.items()\
                                if n not in exclude], 0)
        th0 = torch.quantile(attr_all, 1-k)

    # layer-wise top k activations
    res = {}
    for n,v in attr.items():
        if mode == 'layer':
            th0 = torch.quantile(v, 1-k)
        # masked activations
        res[n] = v >= th0

    return res

def update_mask(masks, new_masks):
    if masks is None:
        return new_masks

    for n,m in masks.items():
        new_m = new_masks[n]
        masks[n] = torch.logical_and(m, new_m)

    return masks