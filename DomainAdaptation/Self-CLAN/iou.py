import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import os
from utils.log import log_message, init_log


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, devkit_dir='', log_file=''):
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
    log_message('Num classes: ' + str(num_classes), log_file)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val.txt')
    label_path_list = join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            log_message('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()), gt_imgs[ind],
                                                                                  pred_imgs[ind]), log_file)
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            log_message('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))), log_file)

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        log_message('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)), log_file)
    log_message('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)), log_file)
    return mIoUs


def main(args):
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = join(args.log_dir, 'result.txt')
    init_log(log_file)
    compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir, log_file)


LABEL_DIR = '/media/data/walteraul_data/datasets/cityscapes/gtFine/val'

EXPERIMENT = '10k_5000'
PRED_DIR = '/media/data/walteraul_data/results/10k_5000/'
LOG_DIR = 'results/10k_5000/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt_dir', type=str, default=LABEL_DIR, help='directory which stores CityScapes val gt images')
    parser.add_argument('-pred_dir', type=str, default=PRED_DIR, help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')
    parser.add_argument('--log_dir', default=LOG_DIR, help='log file directory')

    args = parser.parse_args()
    main(args)
