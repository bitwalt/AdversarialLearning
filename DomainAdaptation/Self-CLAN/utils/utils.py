import os
import sys
import torch
import logging
import argparse
import datetime
import collections
import numpy as np
from PIL import Image, ImageDraw
from collections import OrderedDict

CONFIG = './config/gta_to_cityscapes.yaml'


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-c', '--config', metavar='C', default=CONFIG, help='The Configuration file')
    argparser.add_argument('-e', type=str, help='Experiment name')
    args = argparser.parse_args()
    return args


def rotate_tensor(x, rot):

    if rot == 0:
        img_rt = x
    elif rot == 90:
        img_rt = x.transpose(2, 3)
    elif rot == 180:
        img_rt = x.flip(2)
    elif rot == 270:
        img_rt = x.transpose(2, 3).flip(3)
    else:
        raise ValueError('Rotation angles should be in [0, 90, 180, 270]')
    return img_rt


def cls_acc(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def eval_seg(label_preds, label_trues, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += fast_hist(lp, lt, n_class)

    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iou)
    return mean_iou

def fast_hist(label_pred, label_true, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    return np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2).reshape(n_class, n_class)

def per_class_iou(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def decode_segmap(seg_pred):
    """
    Create RGB images from segmentation predictions.
    """
    pred = seg_pred.data.cpu().numpy()
    pred = np.argmax(pred, axis=0)
    img = mask2color(pred)
    img = np.transpose(img, (2, 0, 1)) # h, w, c -> c, h, w
    return img

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict
