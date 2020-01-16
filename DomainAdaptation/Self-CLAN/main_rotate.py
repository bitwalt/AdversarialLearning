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

from utils.loss import CrossEntropy2d
from utils.loss import WeightedBCEWithLogitsLoss
from utils.log import log_message, init_log
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

from models.aux_model import AuxModel


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'ResNet'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4

IGNORE_LABEL = 255

MOMENTUM = 0.9
NUM_CLASSES = 19

RETRAIN = True
RESTORE_FROM = './model/DeepLab_resnet_pretrained.pth'
# RESTORE_FROM_D = './snapshots/GTA2Cityscapes_CVPR_Syn0820_Wg00005weight005_dampingx2/GTA5_36000_D.pth' #For retrain

###
START_FROM_ITER = 0 #Default 0
####

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 4000

# Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
NUM_STEPS = 50000
NUM_STEPS_STOP = 50000  # Use damping instead of early stopping
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
DATA_LIST_PATH = './dataset/gta5_list/train.txt'

TARGET = 'cityscapes'
INPUT_SIZE_TARGET = [1024, 512]
DATA_DIRECTORY_TARGET = '/media/data/walteraul_data/datasets/cityscapes'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'

SET = 'train'

EXPERIMENT = '50k_ROTATION'

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
    parser.add_argument("--log-dir", type=str, default=LOG_DIR, help="Where to save log of the model.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help="Where restore model parameters from.")
    parser.add_argument("--start-from-iter", type=str, default=START_FROM_ITER, help="Where start model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES, help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY, help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR, help="Where to save snapshots of the model.")
    parser.add_argument("--experiment", type=str, default=EXPERIMENT, help="Experiment name")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--set", type=str, default=SET, help="choose adaptation set.")

    return parser.parse_args()


def loss_calc(pred, label, device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
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

    device = torch.device("cuda" if not args.cpu else "cpu")

    snapshot_dir = os.path.join(args.snapshot_dir, args.experiment)
    os.makedirs(snapshot_dir, exist_ok=True)
    log_file = join(args.log_dir, '%s_log.txt' % args.experiment)

    init_log(log_file, args)

    # =============================================================================
    # INIT G
    # =============================================================================
    model = Res_Deeplab(num_classes=args.num_classes, restore_from=args.restore_from)
    model.train()
    model.to(device)

    # =============================================================================
    # INIT Auxiliary Model
    # =============================================================================
    model_r = AuxModel(config, logger)
    model_r.train()
    model_r.to(device)

    # DataLoaders
    trainloader = data.DataLoader(GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                                            crop_size=args.input_size_source, scale=True, mirror=True, mean=IMG_MEAN),
                                            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size, crop_size=args.input_size_target, scale=True, mirror=True,
                                                     mean=IMG_MEAN, set=args.set), batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=True)
    targetloader_iter = enumerate(targetloader)

# Optimizers
    optimizer = optim.SGD(model.optim_parameters(args), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

# Losses
    bce_loss = torch.nn.BCEWithLogitsLoss()
    weighted_bce_loss = WeightedBCEWithLogitsLoss()

    interp_source = nn.Upsample(size=(args.input_size_source[1], args.input_size_source[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(args.input_size_target[1], args.input_size_target[0]), mode='bilinear', align_corners=True)

    # ======================================================================================
    # Start training
    # ======================================================================================
    print('###########   TRAINING STARTED  ############')
    start = time.time()

    for i_iter in range(args.num_steps-args.start_from_iter):

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter+args.start_from_iter)


    # ======================================================================================
    # train G
    # ======================================================================================

    # Train with Source
        _, batch = next(trainloader_iter)
        images_s, labels_s, _, _ = batch
        images_s = images_s.to(device)
        pred_source1, pred_source2 = model(images_s)

        pred_source1 = interp_source(pred_source1)
        pred_source2 = interp_source(pred_source2)

        # Segmentation Loss
        loss_seg = (loss_calc(pred_source1, labels_s, device) + loss_calc(pred_source2, labels_s, device))
        loss_seg.backward()

    # Get Target prediction
        _, batch = next(targetloader_iter)
        images_t, _, _ = batch
        images_t = images_t.to(device)

        pred_target1, pred_target2 = model(images_t)
        pred_target1 = interp_target(pred_target1)
        pred_target2 = interp_target(pred_target2)

    # Weight Discrepancy Loss
        W5 = None
        W6 = None
        if args.model == 'ResNet':

            for (w5, w6) in zip(model.layer5.parameters(), model.layer6.parameters()):
                if W5 is None and W6 is None:
                    W5 = w5.view(-1)
                    W6 = w6.view(-1)
                else:
                    W5 = torch.cat((W5, w5.view(-1)), 0)
                    W6 = torch.cat((W6, w6.view(-1)), 0)

        loss_weight = (torch.matmul(W5, W6) / (torch.norm(W5) * torch.norm(W6)) + 1)  # +1 is for a positive loss
        loss_weight = loss_weight * Lambda_weight * damping * 2
        loss_weight.backward()

    # ======================================================================================
    # Train SELF SUPERVISED TASK
    # ======================================================================================


        ''' SELF-SUPERVISED (ROTATION) ALGORITHM 
        - Get squared prediction
        - Rotate it randomly (0,90,180,270) -> assign self-label (0,1,2,3)  [*2 IF WANT TO CLASSIFY ALSO S/T]
        - Send rotated prediction to the classifier
        - Get loss 
        - Update weights of classifier and G (segmentation network) 
        '''

        # Train with Source
        pred_source1 = pred_source1.detach()
        pred_source2 = pred_source2.detach()

        # Train with Target
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()

        pred_source = upscale(sum(pred_source1, pred_source2))
        pred_target = upscale(sum(pred_target1, pred_target2))


        loss_rot_source = model_r(pred_source)
        loss_rot_target = model_r(pred_source)

        loss_rot = loss_rot_source+loss_rot_target

        loss_rot.backword()

        optimizer_r.step()
        optimizer.step()


        if (i_iter+args.start_from_iter) % 10 == 0:
            log_message('Iter = {0:6d}/{1:6d}, loss_seg = {2:.4f}, loss_weight = {4:.4f}'.format(i_iter+args.start_from_iter, args.num_steps, loss_seg, loss_weight), log_file)

        if (i_iter+1+args.start_from_iter) % args.save_pred_every == 0 and (i_iter+args.start_from_iter) != 0:
            print('saving weights...')
            torch.save(model.state_dict(), osp.join(snapshot_dir, 'GTA5_' + str(i_iter+args.start_from_iter+1) + '.pth'))
            #save self net

    end = time.time()
    log_message('Total training time: {} days, {} hours, {} min, {} sec '.format(
        int((end - start) / 86400), int((end - start)/3600), int((end - start)/60%60), int((end - start)%60)), log_file)
    print('### Experiment: ' + args.experiment + ' finished ###')

if __name__ == '__main__':
    args = get_arguments()
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system('rm tmp')
    gpu_target = str(np.argmax(memory_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_target
    print('Training on GPU ' + gpu_target)
    main()
