import time, os
import numpy as np
from PIL import Image
from os.path import join
from random import randint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import BCEWithLogitsLoss

from models.CLAN_G import Res_Deeplab

from utils.loss import loss_calc, save_losses_plot
from utils.visual import colorize_mask
from utils.metrics import compute_mIoU
from utils.optimizer import adjust_learning_rate

import random, os
from dataset.data_loader import get_target_train_dataloader, get_target_val_dataloader
from utils.config import process_config, get_args
from utils.log import get_logger

'''
File for training DeepLab on Cityscapes with co-training approach - 
'''

class Deeplab:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_iter = args.start_iter
        self.num_steps = args.num_steps
        self.num_classes = args.num_classes
        self.preheat = self.num_steps / 20  # damping instead of early stopping
        self.source_label = 0
        self.target_label = 1
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.save_path = args.prediction_dir  # dir to save class mIoU when validating model
        self.losses = {'seg_t': list(), 'weight': list()}

        cudnn.enabled = True
        cudnn.benchmark = True

        # set up models
        if args.networks.segmentation == 'DeepLab':
            self.model = Res_Deeplab(num_classes=args.num_classes, restore_from=args.model.restore_from)
            self.optimizer = optim.SGD(self.model.optim_parameters(args.model.optimizer), lr=args.model.optimizer.lr,
                                       momentum=args.model.optimizer.momentum,
                                       weight_decay=args.model.optimizer.weight_decay)

    def train(self, tar_loader, val_loader):

        loss_weight = 0
        args = self.args
        log = self.logger
        device = self.device

        interp_target = nn.Upsample(size=(args.datasets.target.images_size[1], args.datasets.target.images_size[0]),
                                    mode='bilinear', align_corners=True)

        target_iter = enumerate(tar_loader)

        self.model.train()
        self.model = self.model.to(device)

        log.info('###########   TRAINING STARTED  ############')
        start = time.time()

        for i_iter in range(self.start_iter, self.num_steps):

            if i_iter % int(1 / args.target_frac) == 0:
                self.model.train()
                self.optimizer.zero_grad()
                adjust_learning_rate(self.optimizer, self.preheat, args.num_steps, args.power, i_iter, args.model.optimizer)

                damping = (1 - i_iter / self.num_steps)  # similar to early stopping

                # Train with Target
                _, batch = next(target_iter)
                images_t, labels_t = batch
                images_t = images_t.to(device)
                pred_target1, pred_target2 = self.model(images_t)

                pred_target1 = interp_target(pred_target1)
                pred_target2 = interp_target(pred_target2)

                loss_seg_t = (loss_calc(args.num_classes, pred_target1, labels_t, device) + loss_calc(args.num_classes,
                                                                                                          pred_target2,
                                                                                                          labels_t, device))
                loss_seg_t.backward()
                self.losses['seg_t'].append(loss_seg_t.item())

                # Weight Discrepancy Loss
                if args.weight_loss:

                    W5 = None
                    W6 = None
                    # TODO: ADD ERF-NET
                    if args.model.name == 'DeepLab':

                        for (w5, w6) in zip(self.model.layer5.parameters(), self.model.layer6.parameters()):
                            if W5 is None and W6 is None:
                                W5 = w5.view(-1)
                                W6 = w6.view(-1)
                            else:
                                W5 = torch.cat((W5, w5.view(-1)), 0)
                                W6 = torch.cat((W6, w6.view(-1)), 0)

                    # Cosine distance between W5 and W6 vectors
                    loss_weight = (
                                torch.matmul(W5, W6) / (torch.norm(W5) * torch.norm(W6)) + 1)  # +1 is for a positive loss
                    loss_weight = loss_weight * args.Lambda_weight * damping * 2
                    loss_weight.backward()
                    self.losses['weight'].append(loss_weight.item())

                # Optimizers steps
                self.optimizer.step()

                if i_iter % 10 == 0:
                    log.info(
                        'Iter = {0:6d}/{1:6d}, loss_seg = {2:.4f}, loss_weight = {3:.4f}'.format(
                            i_iter, self.num_steps, loss_seg_t, loss_weight))

                if (i_iter % args.save_pred_every == 0 and i_iter != 0) or i_iter == self.num_steps - 1:
                    log.info('saving weights...')
                    i_iter = i_iter if i_iter != self.num_steps - 1 else i_iter + 1  # for last iter
                    torch.save(self.model.state_dict(), join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))

                    self.validate(i_iter, val_loader)
                    compute_mIoU(i_iter, args.datasets.target.val.label_dir, self.save_path, args.datasets.target.json_file,
                                 args.datasets.target.base_list, args.results_dir)
                    #save_losses_plot(args.results_dir, self.losses)

                    # SAVE ALSO IMAGES OF SOURCE AND TARGET
                    #save_segmentations(args.images_dir, images_s, labels_s, pred_source1, images_t)

                del images_t, labels_t, pred_target1, pred_target2

        end = time.time()
        days = int((end - start) / 86400)
        log.info('Total training time: {} days, {} hours, {} min, {} sec '.format(days, int((end - start) / 3600) - (
                    days * 24), int((end - start) / 60 % 60), int((end - start) % 60)))
        print('### Experiment: ' + args.experiment + ' finished ###')

    def validate(self, current_iter, val_loader):
        self.model.eval()
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
        self.save_path = join(self.args.prediction_dir, str(current_iter))
        os.makedirs(self.save_path, exist_ok=True)

        print('### STARTING EVALUATING ###')
        print('total to process: %d' % len(val_loader))
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                if index % 100 == 0:
                    print('%d processed' % index)
                image, _, name, = batch
                output1, output2 = self.model(image.to(self.device))
                output = interp(output1 + output2).cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                output_col = colorize_mask(output)
                output = Image.fromarray(output)

                name = name[0].split('/')[-1]
                output.save('%s/%s' % (self.save_path, name))
                output_col.save('%s/%s_color.png' % (self.save_path, name.split('.')[0]))

        print('### EVALUATING FINISHED ###')


def main():
    args = get_args()
    config = process_config(args.config)

    # logging to the file and stdout
    logger = get_logger(config.log_dir, config.experiment)

    # fix random seed to reproduce results
    random.seed(config.random_seed)
    logger.info('Random seed: {:d}'.format(config.random_seed))

    model = Deeplab(config, logger)

    # Get train dataloader
    target_loader = get_target_train_dataloader(config.datasets.target)

    # Get validation dataloader
    val_loader = get_target_val_dataloader(config.datasets.target)

    if config.mode == 'train':
        model.train(target_loader, val_loader)


if __name__ == '__main__':
    # Use most free gpu
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system('rm tmp')
    gpu_target = str(np.argmax(memory_gpu))
    # gpu_target = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_target
    print('Training on GPU ' + gpu_target)
    main()
