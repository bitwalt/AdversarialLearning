import os
import time
import itertools
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
from utils.optimizer import get_optimizer
from networks import get_auxiliary_net

from models.CLAN_G import Res_Deeplab
from models.Discriminators import Discriminator

from utils.loss import CrossEntropy2d, WeightedBCEWithLogitsLoss
from utils.loss import save_losses_plot
from utils.utils import rotate_tensor
from utils.visual import colorize_mask
from iou import compute_mIoU

class Self_CLAN:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_iter = args.start_iter
        self.num_steps = args.num_steps
        self.num_classes = args.num_classes
        self.preheat = self.num_steps/20 # damping instad of early stopping
        self.source_label = 0 # Labels for Adversarial Training
        self.target_label = 1
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.weighted_bce_loss = WeightedBCEWithLogitsLoss()
        self.save_path = args.prediction_dir
        self.losses = {'seg': list(), 'adv': list(), 'weight': list(), 'ds': list(), 'dt': list(), 'aux': list()}
        self.rotations = [0, 90, 180, 270]

        cudnn.enabled = True
        cudnn.benchmark = True

        # set up models
        if args.networks.segmentation == 'DeepLab':
            self.model = Res_Deeplab(num_classes=args.num_classes, restore_from=args.model.restore_from)
            self.model = self.model.to(self.device)
            optimizer = get_optimizer(args.model.optimizer)
            optimizer_params = {k: v for k, v in args.model.optimizer.items() if k != "name"}
            self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        # ELIF -- ADD HERE NEW SEGMENTATION NET -- ex. ErfNet

        if args.method.adversarial:
            self.model_D = Discriminator(type_d=args.discriminator.type, num_classes=args.num_classes, restore=args.restore, restore_from=args.discriminator.restore_from)
            self.model_D = self.model_D.to(self.device)
            optimizer_D = get_optimizer(args.discriminator.optimizer)
            optimizer_params = {k: v for k, v in args.discriminator.optimizer.items() if k != "name"}
            self.optimizer_D = optimizer_D(self.model_D.parameters(), **optimizer_params)

        if args.method.self:
            self.model_A = get_auxiliary_net(args.auxiliary.name)(input_dim=args.auxiliary.classes, aux_classes=args.auxiliary.n_classes, classes=args.auxiliary.classes, pretrained=False)
            self.model_A = self.model_A.to(self.device)
            optimizer_A = get_optimizer(args.auxiliary.optimizer)
            optimizer_params = {k: v for k, v in args.auxiliary.optimizer.items() if k != "name"}
            self.optimizer_A = optimizer_A(self.model_A.parameters(), **optimizer_params)
            self.aux_loss = nn.CrossEntropyLoss()

    def entropy_loss(self, x):
        return torch.sum(-F.softmax(x, 1) * F.log_softmax(x, 1), 1).mean()

    def weightmap(self, pred1, pred2):
        output = 1.0 - torch.sum((pred1 * pred2), 1).view(1, 1, pred1.size(2), pred1.size(3)) / \
                 (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(1, 1, pred1.size(2), pred1.size(3))
        return output

    def loss_calc(self, pred, label, device):
        """
        This function returns cross entropy loss for semantic segmentation
        """
        # out shape batch_size x channels x h x w -> batch_size x channels x h x w
        # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
        label = label.long().to(device)
        criterion = CrossEntropy2d(self.num_classes).to(device)
        return criterion(pred, label)

    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def lr_warmup(self, base_lr, iter, warmup_iter):
        return base_lr * (float(iter) / warmup_iter)

    def adjust_learning_rate(self, optimizer, i_iter, args):
        if i_iter < self.preheat:
            lr = self.lr_warmup(args.optimizer.lr, i_iter, self.preheat)
        else:
            lr = self.lr_poly(args.optimizer.lr, i_iter, self.num_steps, self.args.power)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10


    def train(self, src_loader, tar_loader, val_loader):

        loss_rot = loss_adv = loss_weight = loss_D_s = loss_D_t = 0
        args = self.args
        log = self.logger

        interp_source = nn.Upsample(size=(args.datasets.source.images_size[1], args.datasets.source.images_size[0]), mode='bilinear',  align_corners=True)
        interp_target = nn.Upsample(size=(args.datasets.target.images_size[1], args.datasets.target.images_size[0]), mode='bilinear',  align_corners=True)
        interp_prediction = nn.Upsample(size=(args.auxiliary.images_size[1], args.auxiliary.images_size[0]), mode='bilinear', align_corners=True)

        source_iter = enumerate(src_loader)
        target_iter = enumerate(tar_loader)

        self.model.train()

        if args.method.adversarial:
            self.model_D.train()
        if args.method.self:
            self.model_A.train()

        device = self.device

        log.info('###########   TRAINING STARTED  ############')
        start = time.time()

        for i_iter in range(self.start_iter, self.num_steps):

            self.model.train()
            self.optimizer.zero_grad()
            self.adjust_learning_rate(self.optimizer, i_iter, args.model)

            if args.method.adversarial:
                self.optimizer_D.zero_grad()
                self.adjust_learning_rate(self.optimizer_D, i_iter, args.discriminator)
                self.model_D.train()
            if args.method.self:
                self.optimizer_A.zero_grad()
                self.adjust_learning_rate(self.optimizer_A, i_iter, args.auxiliary)
                self.model_A.train()

            damping = (1 - i_iter/self.num_steps) #similar to early stopping

        # ======================================================================================
        # train G
        # ======================================================================================
            if args.method.adversarial:
            # Remove Grads in D
                for param in self.model_D.parameters():
                    param.requires_grad = False

            # Train with Source
            _, batch = next(source_iter)
            images_s, labels_s, _, _ = batch
            images_s = images_s.to(device)
            pred_source1_, pred_source2_ = self.model(images_s)

            pred_source1 = interp_source(pred_source1_)
            pred_source2 = interp_source(pred_source2_)

        # Segmentation Loss
            loss_seg = (self.loss_calc(pred_source1, labels_s, device) + self.loss_calc(pred_source2, labels_s, device))
            loss_seg.backward()
            self.losses['seg'].append(loss_seg.item())

            if not args.solo_source:
                _, batch = next(target_iter)
                images_t, _ = batch # get target label here
                images_t = images_t.to(device)
                pred_target1_, pred_target2_ = self.model(images_t)
            # TODO: ADD SEGMENTATION LOSS ALSO X TARGET -> semi-supervised approach

        # Adversarial Loss
            if args.method.adversarial:
                # Train with Target
                #_, batch = next(target_iter)
                #images_t, _ = batch
                #images_t = images_t.to(device)
                #pred_target1, pred_target2 = self.model(images_t)

                pred_target1 = interp_target(pred_target1_)
                pred_target2 = interp_target(pred_target2_)

                weight_map = self.weightmap(F.softmax(pred_target1, dim=1), F.softmax(pred_target2, dim=1))

                D_out = interp_target(self.model_D(F.softmax(pred_target1 + pred_target2, dim=1)))

                # Adaptive Adversarial Loss
                if i_iter > self.preheat:
                    loss_adv = self.weighted_bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(self.source_label).to(device),
                                                 weight_map, args.Epsilon, args.Lambda_local)
                else:
                    loss_adv = self.bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(self.source_label).to(device))

                loss_adv = loss_adv * self.args.Lambda_adv * damping
                loss_adv.backward()
                self.losses['adv'].append(loss_adv.item())

        # Weight Discrepancy Loss
            if args.weight_loss:
                W5 = None
                W6 = None
                if args.model.name == 'DeepLab':

                    for (w5, w6) in zip(self.model.layer5.parameters(), self.model.layer6.parameters()):
                        if W5 is None and W6 is None:
                            W5 = w5.view(-1)
                            W6 = w6.view(-1)
                        else:
                            W5 = torch.cat((W5, w5.view(-1)), 0)
                            W6 = torch.cat((W6, w6.view(-1)), 0)

                loss_weight = (torch.matmul(W5, W6) / (torch.norm(W5) * torch.norm(W6)) + 1)  # +1 is for a positive loss
                loss_weight = loss_weight * args.Lambda_weight * damping * 2
                loss_weight.backward()
                self.losses['weight'].append(loss_weight.item())

        # ======================================================================================
        # train D
        # ======================================================================================
            if args.method.adversarial:
                # Bring back Grads in D
                for param in self.model_D.parameters():
                    param.requires_grad = True

                # Train with Source
                pred_source1 = pred_source1.detach()
                pred_source2 = pred_source2.detach()

                D_out_s = interp_source(self.model_D(F.softmax(pred_source1 + pred_source2, dim=1)))

                loss_D_s = self.bce_loss(D_out_s, torch.FloatTensor(D_out_s.data.size()).fill_(self.source_label).to(device))

                loss_D_s.backward()
                self.losses['ds'].append(loss_D_s.item())

                # Train with Target
                pred_target1 = pred_target1.detach()
                pred_target2 = pred_target2.detach()
                weight_map = weight_map.detach()

                D_out_t = interp_target(self.model_D(F.softmax(pred_target1 + pred_target2, dim=1)))

                # Adaptive Adversarial Loss
                if (i_iter > self.preheat):
                    loss_D_t = self.weighted_bce_loss(D_out_t, torch.FloatTensor(D_out_t.data.size()).fill_(self.target_label).to(device),
                                                 weight_map, args.Epsilon, args.Lambda_local)
                else:
                    loss_D_t = self.bce_loss(D_out_t, torch.FloatTensor(D_out_t.data.size()).fill_(self.target_label).to(device))

                loss_D_t.backward()
                self.losses['dt'].append(loss_D_t.item())

        # ======================================================================================
        # Train SELF SUPERVISED TASK
        # ======================================================================================
            if args.method.self:

                ''' SELF-SUPERVISED (ROTATION) ALGORITHM 
                - Get squared prediction 
                - Rotate it randomly (0,90,180,270) -> assign self-label (0,1,2,3)  [*2 IF WANT TO CLASSIFY ALSO S/T]
                - Send rotated prediction to the classifier
                - Get loss 
                - Update weights of classifier and G (segmentation network) 
                '''

                #TODO: GET RANDOM PREDICTION
                # Train with Source
                pred_source1 = pred_source1_.detach()
                pred_source2 = pred_source2_.detach()

                # Train with Target
                pred_target1 = pred_target1_.detach()
                pred_target2 = pred_target2_.detach()

                # save_prediction(pred_source1, './temp/before_square_color.png')
                pred_source = interp_prediction(F.softmax(pred_source1 + pred_source2, dim=1))
                pred_target = interp_prediction(F.softmax(pred_target1 + pred_target2, dim=1))

                # save_prediction(pred_source, './temp/square_color.png')

            # ROTATE TENSOR
                # source
                label_source = torch.empty(1, dtype=torch.long).random_(args.auxiliary.n_classes).to(device)
                rotated_pred_source = rotate_tensor(pred_source, self.rotations[label_source.item()])
                # save_prediction(pred_source, './temp/square_rotated_prediction_color.png')
                pred_source_label = self.model_A(rotated_pred_source)
                loss_rot_source = self.aux_loss(pred_source_label, label_source)

                # target
                label_target = torch.empty(1, dtype=torch.long).random_(args.auxiliary.n_classes).to(device)
                rotated_pred_target = rotate_tensor(pred_target, self.rotations[label_target.item()])
                pred_target_label = self.model_A(rotated_pred_target)
                loss_rot_target = self.aux_loss(pred_target_label, label_target)

                loss_rot = (loss_rot_source + loss_rot_target) * args.Lambda_aux

                loss_rot.backward()
                self.losses['aux'].append(loss_rot.item())

            # Optimizers steps
            self.optimizer.step()
            if args.method.adversarial:
                self.optimizer_D.step()
            if args.method.self:
                self.optimizer_A.step()

            if i_iter % 10 == 0:
                log.info('Iter = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_rot = {3:.4f}, loss_adv = {4:.4f}, loss_weight = {5:.4f}, loss_D_s = {6:.4f} loss_D_t = {7:.4f}'.format(
                        i_iter, self.num_steps, loss_seg, loss_rot, loss_adv, loss_weight, loss_D_s, loss_D_t))

            if (i_iter % args.save_pred_every == 0 and i_iter != 0) or i_iter == self.num_steps-1:
                log.info('saving weights...')
                it = i_iter if i_iter != self.num_steps-1 else i_iter+1 # for last iter
                torch.save(self.model.state_dict(), join(args.snapshot_dir, 'GTA5_' + str(it) + '.pth'))
                if args.method.adversarial:
                    torch.save(self.model_D.state_dict(), join(args.snapshot_dir, 'GTA5_' + str(it) + '_D.pth'))
                if args.method.self:
                    torch.save(self.model_A.state_dict(), join(args.snapshot_dir, 'GTA5_' + str(it) + '_Aux.pth'))

                self.save_path = join(self.args.prediction_dir, str(i_iter))
                os.makedirs(self.save_path, exist_ok=True)

                self.validate(i_iter, val_loader)
                compute_mIoU(i_iter, args.datasets.target.val.label_dir, self.save_path, args.datasets.target.json_file, args.datasets.target.base_list, args.results_dir)

        save_losses_plot(args.results_dir, self.losses)
        end = time.time()
        days = int((end - start) / 86400)
        log.info('Total training time: {} days, {} hours, {} min, {} sec '.format(
            days, int((end - start) / 3600)-(days*24), int((end - start) / 60 % 60),  int((end - start) % 60)))
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
