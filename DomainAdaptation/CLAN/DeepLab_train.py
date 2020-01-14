import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import time, timeit, datetime
from model.CLAN_G import Res_Deeplab
from model.CLAN_D import FCDiscriminator

from utils.loss import CrossEntropy2d
from utils.loss import WeightedBCEWithLogitsLoss
from utils.log import log_message, init_log
from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SYNTHIADataSet
from dataset.cityscapes_dataset import cityscapesDataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'ResNet'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4

IGNORE_LABEL = 255

MOMENTUM = 0.9
NUM_CLASSES = 19
RESTORE_FROM = './model/DeepLab_resnet_pretrained_init-f81d91e8.pth'
# RESTORE_FROM_D = './snapshots/GTA2Cityscapes_CVPR_Syn0820_Wg00005weight005_dampingx2/GTA5_36000_D.pth' #For retrain

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2000

# Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
NUM_STEPS = 1
NUM_STEPS_STOP = 10000  # Use damping instead of early stopping
PREHEAT_STEPS = int(NUM_STEPS_STOP / 20)
POWER = 0.9
RANDOM_SEED = 1234
Lambda_weight = 0.01
Lambda_adv = 0.001
Lambda_local = 40
Epsilon = 0.4

SOURCE = 'GTA5'
INPUT_SIZE_SOURCE = [1280, 720]
DATA_DIRECTORY = '/media/data/walteraul_data/datasets/gta5'
DATA_LIST_PATH = './dataset/gta5_list/train5000.txt'

TARGET = 'cityscapes'
INPUT_SIZE_TARGET = [1024, 512]
DATA_DIRECTORY_TARGET = '/media/data/walteraul_data/datasets/cityscapes'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'

SET = 'train'


SNAPSHOT_DIR = './snapshots/Source_Segmentation'
LOG_DIR = './log'
LOG_FILE = 'log/Source_Segmentation.txt'  # INSERT EXPERIMENT NAME HERE


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL, help="available options : ResNet")
    parser.add_argument("--source", type=str, default=SOURCE, help="available options : GTA5, SYNTHIA")
    parser.add_argument("--target", type=str, default=TARGET, help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE, help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY, help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH, help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL, help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size-source", type=str, default=INPUT_SIZE_SOURCE, help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET, help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET, help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET, help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true", help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D, help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true", help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES, help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS, help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP, help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER, help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true", help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true", help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES, help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR, help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET, help="choose adaptation set.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR, help="Path to the directory of log.")

    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d(NUM_CLASSES).cuda(gpu)
    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate_D, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def weightmap(pred1, pred2):
    output = 1.0 - torch.sum((pred1 * pred2), 1).view(1, 1, pred1.size(2), pred1.size(3)) / \
             (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(1, 1, pred1.size(2), pred1.size(3))
    return output


def main():
    """Create the model and start the training."""

    cudnn.enabled = True
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # =============================================================================
    # INIT G
    # =============================================================================
    if MODEL == 'ResNet':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http':
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]

        if args.restore_from[:4] == './mo':
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)

    model.train()
    model.cuda(args.gpu)

    # DataLoaders
    trainloader = data.DataLoader(GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                                            crop_size=args.input_size_source, scale=True, mirror=True, mean=IMG_MEAN),
                                            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)

    # targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
    #                                                  max_iters=args.num_steps * args.iter_size * args.batch_size, crop_size=args.input_size_target, scale=True, mirror=True,
    #                                                  mean=IMG_MEAN, set=args.set), batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=True)
    # targetloader_iter = enumerate(targetloader)

    # Optimizers
    optimizer = optim.SGD(model.optim_parameters(args), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()


    interp_source = nn.Upsample(size=(args.input_size_source[1], args.input_size_source[0]), mode='bilinear', align_corners=True)
    # interp_target = nn.Upsample(size=(args.input_size_target[1], args.input_size_target[0]), mode='bilinear', align_corners=True)

    # ======================================================================================
    # Start training
    # ======================================================================================
    log_message('###########   TRAINING STARTED  ############', LOG_FILE)
    start = time.time()

    for i_iter in range(args.num_steps):

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        # ======================================================================================
        # train G
        # ======================================================================================

        # Remove Grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # Train with Source
        _, batch = next(trainloader_iter)
        images_s, labels_s, _, _ = batch
        images_s = Variable(images_s).cuda(args.gpu)
        pred_source1, pred_source2 = model(images_s)

        pred_source1 = interp_source(pred_source1)
        pred_source2 = interp_source(pred_source2)

        # Segmentation Loss
        loss_seg = (loss_calc(pred_source1, labels_s, args.gpu) + loss_calc(pred_source2, labels_s, args.gpu))
        loss_seg.backward()

        optimizer.step()

        if i_iter % 10 == 0:
            log_message('Iter = {0:6d}/{1:6d}, loss_seg = {2:.4f}'.format(i_iter, args.num_steps, loss_seg), LOG_FILE)

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))

    end = time.time()
    log_message('Total training time: {} days, {} hours, {} min, {} sec '.format(
        int((end - start) / 86400), int((end - start) / 3600), int((end - start) / 60), int((end - start) % 60)), LOG_FILE)


if __name__ == '__main__':
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system('rm tmp')
    gpu_target = str(np.argmax(memory_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_target
    init_log(LOG_FILE)
    log_message('Training on GPU ' + gpu_target, LOG_FILE)
    main()
