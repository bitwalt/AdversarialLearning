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
from dataset.cityscapes_dataset_label import cityscapesDataSetLabel

from dataset.cityscapes import Cityscapes

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'ResNet'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4

IGNORE_LABEL = 255

MOMENTUM = 0.9
NUM_CLASSES = 19
RESTORE_FROM = './model/DeepLab_resnet_pretrained.pth'

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 4000

# Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
NUM_STEPS = 48001
NUM_STEPS_STOP = 48001  # Use damping instead of early stopping
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
DATA_LIST_PATH = './dataset/gta5_list/train10000.txt'

TARGET = 'cityscapes'
INPUT_SIZE_TARGET = [1024, 512]
DATA_DIRECTORY_TARGET = '/media/data/walteraul_data/datasets/cityscapes'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
DATA_LABEL_PATH='/media/data/walteraul_data/datasets/cityscapes/gtFine/train'
DATA_LIST_LABEL_TARGET = './dataset/cityscapes_list/train_label.txt'
SET = 'train'

EXPERIMENT = 'DeepLab_Cityscapes'
SNAPSHOT_DIR = '/media/data/walteraul_data/snapshots/'
LOG_DIR = 'log'

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
    parser.add_argument("--data-list-label-target", type=str, default=DATA_LIST_LABEL_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--label-dir", type=str, default=DATA_LABEL_PATH, help="Path to the file listing the images in the target dataset.")

    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET, help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true", help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Base learning rate for training with polynomial decay.")
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
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY, help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR, help="Where to save snapshots of the model.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR, help="Where to save log of the model.")
    parser.add_argument("--experiment", type=str, default=EXPERIMENT, help="Experiment name")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--set", type=str, default=SET, help="choose adaptation set.")

    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    #label = label.long().to(device)
    label = label.long().to(device)
    criterion = CrossEntropy2d(NUM_CLASSES).to(device)
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


def main():
    """Create the model and start the training."""

    cudnn.enabled = True
    cudnn.benchmark = True

    device = torch.device("cuda" if not args.cpu else "cpu")

    snapshot_dir = os.path.join(args.snapshot_dir, args.experiment)
    os.makedirs(snapshot_dir, exist_ok=True)

    log_file = os.path.join(args.log_dir, '%s.txt' % args.experiment)
    init_log(log_file, args)

    # =============================================================================
    # INIT G
    # =============================================================================
    if MODEL == 'ResNet':
        model = Res_Deeplab(num_classes=args.num_classes, restore_from=args.restore_from)
    model.train()
    model.to(device)

    # DataLoaders
 #  trainloader = data.DataLoader(GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
 #                                         crop_size=args.input_size_source, scale=True, mirror=True, mean=IMG_MEAN),
 #                                          batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
 #   trainloader_iter = enumerate(trainloader)

    trainloader = data.DataLoader(cityscapesDataSetLabel(args.data_dir_target, './dataset/cityscapes_list/info.json', args.data_list_target,args.data_list_label_target,
                                                    max_iters=args.num_steps * args.iter_size * args.batch_size, crop_size=args.input_size_target,
                                                    mean=IMG_MEAN, set=args.set), batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    # Optimizers
    optimizer = optim.SGD(model.optim_parameters(args), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    #interp_source = nn.Upsample(size=(args.input_size_source[1], args.input_size_source[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(args.input_size_target[1], args.input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # ======================================================================================
    # Start training
    # ======================================================================================
    log_message('###########   TRAINING STARTED  ############', log_file)
    start = time.time()

    for i_iter in range(args.num_steps):

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        # ======================================================================================
        # train G
        # ======================================================================================

        # Train with Source
        _, batch = next(trainloader_iter)
        images_s, labels_s = batch
        images_s = images_s.to(device)

        pred_source1, pred_source2 = model(images_s)

        pred_source1 = interp_target(pred_source1)
        pred_source2 = interp_target(pred_source2)
        # Segmentation Loss
        loss_seg = (loss_calc(pred_source1, labels_s, device) + loss_calc(pred_source2, labels_s, device))
        loss_seg.backward()

        optimizer.step()

        if i_iter % 10 == 0:
            log_message('Iter = {0:6d}/{1:6d}, loss_seg = {2:.4f}'.format(i_iter, args.num_steps-1, loss_seg), log_file)

        if (i_iter % args.save_pred_every == 0 and i_iter != 0) or i_iter == args.num_steps-1:
            print('saving weights...')
            torch.save(model.state_dict(), osp.join(snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))

    end = time.time()
    log_message('Total training time: {} days, {} hours, {} min, {} sec '.format(
        int((end - start) / 86400), int((end - start)/3600), int((end - start)/60%60), int((end - start)%60)), log_file)
    print('### Experiment: ' + args.experiment + ' Finished ###')

if __name__ == '__main__':
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system('rm tmp')
    gpu_target = str(np.argmax(memory_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_target
    print('Training on GPU ' + gpu_target)
    main()
