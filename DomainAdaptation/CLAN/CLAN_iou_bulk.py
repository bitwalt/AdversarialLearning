import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import csv
import os


LABEL_DIR = '/media/data/walteraul_data/datasets/cityscapes/gtFine/val'
PRED_DIR = '/media/data/walteraul_data/results/'
LOG_DIR = 'mIoU_results/'

###
EXPERIMENT = '20k_5000GTA'
###
SAVE_STEP = 4000



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


def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
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
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return str(round(np.nanmean(mIoUs) * 100, 2))


def main(args):

    pred_dir = join(args.pred_dir, args.experiment)
    log_file = join(args.log_dir, '%s.txt' % args.experiment)

    n_files = len([name for name in os.listdir(pred_dir)])

    with open(log_file, "w+", newline="") as file:
        for i in range(1, n_files+1):
            print('### Scoring prediction ' + str(i) + '/' + str(n_files) + ' ###')
            pred_i_dir = join(pred_dir, '{0:d}'.format(i * args.save_step))
            mIoU = compute_mIoU(args.gt_dir, pred_i_dir, args.devkit_dir)
            file.write('step_{0:d}'.format(i * args.save_step) + '\t\t===> mIoU: ' + mIoU + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt_dir', type=str, default=LABEL_DIR, help='directory which stores CityScapes val gt images')
    parser.add_argument('-pred_dir', type=str, default=PRED_DIR, help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')
    parser.add_argument('--log_dir', default=LOG_DIR, help='log file directory')
    parser.add_argument("--experiment", type=str, default=EXPERIMENT, help="Experiment name")
    parser.add_argument("--save_step", type=str, default=SAVE_STEP, help="Number of iter for each checkpoint")
    args = parser.parse_args()

    main(args)