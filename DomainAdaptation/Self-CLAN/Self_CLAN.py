
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
from models.networks.alexnet import alexnet
from models.networks.auxiliary import auxiliary

from models.CLAN_G import Res_Deeplab
from models.erfnet_imagenet import ERFNet
from models.Discriminators import discriminator

from utils.loss import WeightedBCEWithLogitsLoss, loss_calc, save_losses_plot
from utils.visual import colorize_mask, weightmap, rotate_tensor, save_segmentations, save_rotations
from utils.metrics import compute_mIoU, AverageMeter
from utils.optimizer import adjust_learning_rate


class Self_CLAN:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_iter = args.start_iter
        self.num_steps = args.num_steps
        self.num_classes = args.num_classes
        self.preheat = self.num_steps/20  # damping instead of early stopping
        self.source_label = 0
        self.target_label = 1
        self.best_miou = 0
        # TODO: CHANGE LOSS for LSGAN
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        #self.bce_loss = torch.nn.MSELoss() #LSGAN
        self.weighted_bce_loss = WeightedBCEWithLogitsLoss()
        self.aux_acc = AverageMeter()
        self.save_path = args.prediction_dir  # dir to save class mIoU when validating model
        self.losses = {'seg': list(),'seg_t': list(), 'adv': list(), 'weight': list(), 'ds': list(), 'dt': list(), 'aux': list()}
        self.rotations = [0, 90, 180, 270]

        cudnn.enabled = True
        #cudnn.benchmark = True

        # set up models
        if args.model.name == 'DeepLab':
            self.model = Res_Deeplab(num_classes=args.num_classes, restore_from=args.model.restore_from)
            self.optimizer = optim.SGD(self.model.optim_parameters(args.model.optimizer), lr=args.model.optimizer.lr, momentum=args.model.optimizer.momentum, weight_decay=args.model.optimizer.weight_decay)
        if args.model.name == 'ErfNet':
            self.model = ERFNet(args.num_classes)  # To add image-net pre-training and double classificator
            self.optimizer = optim.SGD(self.model.optim_parameters(args.model.optimizer), lr=args.model.optimizer.lr, momentum=args.model.optimizer.momentum, weight_decay=args.model.optimizer.weight_decay)

        if args.method.adversarial:
            self.model_D = discriminator(name=args.discriminator.name, num_classes=args.num_classes, restore_from=args.discriminator.restore_from)
            self.optimizer_D = optim.Adam(self.model_D.parameters(), lr=args.discriminator.optimizer.lr, betas=(0.9, 0.99))
        if args.method.self:
            self.model_A = auxiliary(name=args.auxiliary.name, input_dim=args.auxiliary.classes, aux_classes=args.auxiliary.aux_classes, restore_from=args.auxiliary.restore_from)
            self.optimizer_A = optim.Adam(self.model_A.parameters(), lr=args.auxiliary.optimizer.lr, betas=(0.9, 0.99))
            self.aux_loss = nn.CrossEntropyLoss()

    def train(self, src_loader, tar_loader, val_loader):

        loss_rot = loss_adv = loss_weight = loss_D_s = loss_D_t = 0
        args = self.args
        log = self.logger
        device = self.device

        interp_source = nn.Upsample(size=(args.datasets.source.images_size[1], args.datasets.source.images_size[0]), mode='bilinear',  align_corners=True)
        interp_target = nn.Upsample(size=(args.datasets.target.images_size[1], args.datasets.target.images_size[0]), mode='bilinear',  align_corners=True)
        interp_prediction = nn.Upsample(size=(args.auxiliary.images_size[1], args.auxiliary.images_size[0]), mode='bilinear', align_corners=True)

        source_iter = enumerate(src_loader)
        target_iter = enumerate(tar_loader)

        self.model.train()
        self.model = self.model.to(device)

        if args.method.adversarial:
            self.model_D.train()
            self.model_D = self.model_D.to(device)

        if args.method.self:
            self.model_A.train()
            self.model_A = self.model_A.to(device)

        log.info('###########   TRAINING STARTED  ############')
        start = time.time()

        for i_iter in range(self.start_iter, self.num_steps):

            self.model.train()
            self.optimizer.zero_grad()
            adjust_learning_rate(self.optimizer, self.preheat, args.num_steps, args.power, i_iter, args.model.optimizer)

            # Train with adversarial loss
            if args.method.adversarial:
                self.model_D.train()
                self.optimizer_D.zero_grad()
                adjust_learning_rate(self.optimizer_D, self.preheat, args.num_steps, args.power, i_iter, args.discriminator.optimizer)

            # Adding Rotation task
            if args.method.self:
                self.model_A.train()
                self.optimizer_A.zero_grad()
                adjust_learning_rate(self.optimizer_A, self.preheat, args.num_steps, args.power, i_iter, args.auxiliary.optimizer)

            damping = (1 - i_iter/self.num_steps)  # similar to early stopping

        # ======================================================================================
        # train G
        # ======================================================================================
            if args.method.adversarial:
                for param in self.model_D.parameters():  # Remove Grads in D
                    param.requires_grad = False

            # Train with Source
            _, batch = next(source_iter)
            images_s, labels_s, _, _ = batch
            images_s = images_s.to(device)
            pred_source1_, pred_source2_ = self.model(images_s)

            pred_source1 = interp_source(pred_source1_)
            pred_source2 = interp_source(pred_source2_)

            # Segmentation Loss
            loss_seg = (loss_calc(self.num_classes, pred_source1, labels_s, device) + loss_calc(self.num_classes, pred_source2, labels_s, device))
            loss_seg.backward()
            self.losses['seg'].append(loss_seg.item())

            # Train with Target
            _, batch = next(target_iter)
            images_t, labels_t = batch
            images_t = images_t.to(device)
            pred_target1_, pred_target2_ = self.model(images_t)

            pred_target1 = interp_target(pred_target1_)
            pred_target2 = interp_target(pred_target2_)

            # Semi-supervised approach
            if args.use_target_labels and i_iter % int(1 / args.target_frac) == 0:
                loss_seg_t = (loss_calc(args.num_classes, pred_target1, labels_t, device) + loss_calc(args.num_classes, pred_target2, labels_t, device))
                loss_seg_t.backward()
                self.losses['seg_t'].append(loss_seg_t.item())

            # Adversarial Loss
            if args.method.adversarial:

                pred_target1 = pred_target1.detach()
                pred_target2 = pred_target2.detach()

                # TODO: Save the weightmap
                weight_map = weightmap(F.softmax(pred_target1, dim=1), F.softmax(pred_target2, dim=1))

                D_out = interp_target(self.model_D(F.softmax(pred_target1 + pred_target2, dim=1)))

                # Adaptive Adversarial Loss
                if i_iter > self.preheat:
                    loss_adv = self.weighted_bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(self.source_label).to(device), weight_map, args.Epsilon, args.Lambda_local)
                else:
                    loss_adv = self.bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(self.source_label).to(device))

                loss_adv.requires_grad = True
                loss_adv = loss_adv * self.args.Lambda_adv * damping
                loss_adv.backward()
                self.losses['adv'].append(loss_adv.item())

        # Weight Discrepancy Loss
            if args.weight_loss:

                # Init container variables of DeepLab weights of layers 5 and 6
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
                if i_iter > self.preheat:
                    loss_D_t = self.weighted_bce_loss(D_out_t, torch.FloatTensor(D_out_t.data.size()).fill_(self.target_label).to(device), weight_map, args.Epsilon, args.Lambda_local)
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

                # Train with Target
                pred_target1 = pred_target1_.detach()
                pred_target2 = pred_target2_.detach()

                pred_target = interp_prediction(F.softmax(pred_target1 + pred_target2, dim=1))

                # Rotate prediction randomly
                label_target = torch.empty(1, dtype=torch.long).random_(args.auxiliary.aux_classes).to(device)
                rotated_pred_target = rotate_tensor(pred_target, self.rotations[label_target.item()])

                pred_target_label = self.model_A(rotated_pred_target)
                loss_rot_target = self.aux_loss(pred_target_label, label_target)

                save_rotations(args.images_dir, pred_target, rotated_pred_target, i_iter)

                # calculate accuracy of aux
                if label_target.item() == F.softmax(pred_target_label, dim=1).argmax(dim=1).item():
                    self.aux_acc.update(1)
                else:
                    self.aux_acc.update(0)

                loss_rot = loss_rot_target * args.Lambda_aux
                loss_rot.backward()
                self.losses['aux'].append(loss_rot.item())

            # Optimizers steps
            self.optimizer.step()
            if args.method.adversarial:
                self.optimizer_D.step()
            if args.method.self:
                self.optimizer_A.step()

            if i_iter % 10 == 0:
                log.info('Iter = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_rot = {3:.4f}, loss_adv = {4:.4f}, loss_weight = {5:.4f}, loss_D_s = {6:.4f} loss_D_t = {7:.4f}, aux_acc = {8:.2f}%'.format(
                        i_iter, self.num_steps, loss_seg, loss_rot, loss_adv, loss_weight, loss_D_s, loss_D_t, self.aux_acc.val))

            if (i_iter % args.save_pred_every == 0 and i_iter != 0) or i_iter == self.num_steps-1:
                i_iter = i_iter if i_iter != self.num_steps - 1 else i_iter + 1  # for last iter

                # Validate and calculate mIoU
                self.validate(i_iter, val_loader)
                miou = compute_mIoU(i_iter, args.datasets.target.val.label_dir, self.save_path, args.datasets.target.json_file,
                            args.datasets.target.base_list, args.results_dir)

                log.info('saving weights...')

                # TODO: SAVE ONLY BEST AND LAST MODELS
                torch.save(self.model.state_dict(), join(args.snapshot_dir, 'GTA5' + '.pth'))
                if args.method.adversarial:
                    torch.save(self.model_D.state_dict(), join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pth'))
                if args.method.self:
                    torch.save(self.model_A.state_dict(), join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_Aux.pth'))

                # SAVE LOSS PLOT, EXAMPLE TENSORS
                save_losses_plot(args.results_dir, self.losses)
                # TODO: SAVE TENSORS ALSO FOR ADV AND SELF
                save_segmentations(args.images_dir, images_s, labels_s, pred_source1, images_t)
                save_rotations(args.images_dir, pred_target, rotated_pred_target, i_iter)

            del images_s, labels_s, pred_source1, pred_source2, pred_source1_, pred_source2_
            del images_t, labels_t, pred_target1, pred_target2, pred_target1_, pred_target2_, rotated_pred_target

        end = time.time()
        days = int((end - start) / 86400)
        log.info('Total training time: {} days, {} hours, {} min, {} sec '.format(days, int((end - start) / 3600)-(days*24), int((end - start) / 60 % 60),  int((end - start) % 60)))
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
