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
from itertools import pairwise, combinations
from torch.utils.data import DataLoader
from torch.nn.utils import prune
import numpy as np
import torch

def get_attr(model, data_loader, attr_method, transform=None, target_loader=None, get_y=False, device='cpu'):
    attributions = {}

    # aggregate attribution for each layer
    for n,l in model.named_children():
        res = []
        attr = eval(attr_method)(model, l)
        #attr = eval(attr_method)(model, l, multiply_by_inputs=False)

        # setup data loader
        if target_loader:
            loader = zip(data_loader, target_loader)
        else:
            loader = data_loader

        # load data
        for (x,y) in loader:
            x = transform(x) if transform else x        # apply transform
            x, y = x.to(device), y.to(device)

            if get_y:
                with torch.no_grad():
                    _,y = model(x).max(1)

            if attr_method == 'LayerActivation':
                a = attr.attribute(x)

            elif attr_method in ['LayerConductance',
                                 'InternalInfluence',
                                 'LayerIntegratedGradients']:
                a = attr.attribute(x, target=y, n_steps=20)

            elif attr_method in ['LayerDeepLiftShap',
                                 'LayerGradientShap']:
                b = torch.zeros_like(x)
                a = attr.attribute(x, target=y, baselines=b)

            else:
                a = attr.attribute(x, target=y)

            # append result
            res.append(a.detach().cpu())

        res = torch.cat(res, 0)
        attributions[n] = res

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

            th0 = torch.quantile(atr_diff, 1-k)
            masks[n] = atr_diff <= th0

    return masks


def get_mask_rand(norm_attr, adv_attr, k, exclude=[]):
    masks = {}
    for n in norm_attr.keys():
        atr_n = norm_attr[n]
        if n in exclude or k == 0:
            masks[n] = torch.ones(atr_n.size()[1:], dtype=bool)
        else:
            masks[n] = torch.rand(atr_n.size()[1:]) > k

    return masks


# Gyumin's method
def get_mask_top(norm_attr, adv_attr, k, exclude=[]):
    masks = {}
    for n in norm_attr.keys():
        # get attributions
        atr_n = norm_attr[n]
        atr_a = adv_attr[n]

        if n in exclude or k == 0:
            masks[n] = torch.ones(atr_n.size()[1:], dtype=bool)
        else:
            # threshold on zero
            atr_n = (atr_n > 0).type(torch.float)
            atr_a = (atr_a > 0).type(torch.float)

            # count frequencies
            cnt_n = torch.mean(atr_n, 0)
            cnt_a = torch.mean(atr_a, 0)

            # threshold by quantile
            top_n = torch.quantile(cnt_n, 1-k, interpolation='nearest')
            top_a = torch.quantile(cnt_a, 1-k, interpolation='nearest')

            # top k neurons in each category
            neu_n = cnt_n > top_n
            neu_a = cnt_a > top_a

            # only mask
            masks[n] = torch.logical_or(neu_n, ~neu_a)

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