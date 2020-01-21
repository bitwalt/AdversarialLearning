import os
import time
import itertools
import numpy as np
from PIL import Image
from tqdm import tqdm
from random import randint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.nn import BCEWithLogitsLoss
from optimizers import get_optimizer
from networks import get_auxiliary_net
from utils.metrics import AverageMeter
from utils.utils import to_device

from models.CLAN_G import Res_Deeplab
from models.CLAN_D import Discriminator

from utils.loss import CrossEntropy2d
from utils.loss import WeightedBCEWithLogitsLoss

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def save_prediction(tensor, file):
    output = tensor.cpu().data[0].numpy()
    output = output.transpose(1, 2, 0)
    output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
    output_col = colorize_mask(output)
    # output = Image.fromarray(output)
    output_col.save(file)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

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

class CLAN:
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
        self.seg_loss = []
        self.weight_loss = []
        self.adv_loss = []
        self.aux_loss = []
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

        args = self.args

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

        print('###########   TRAINING STARTED  ############')
        start = time.time()

        for i_iter in range(self.start_iter, self.num_steps):

            self.optimizer.zero_grad()
            self.adjust_learning_rate(self.optimizer, i_iter, args.model)

            if args.method.adversarial:
                self.optimizer_D.zero_grad()
                self.adjust_learning_rate(self.optimizer_D, i_iter, args.discriminator)

            if args.method.self:
                self.optimizer_A.zero_grad()
                self.adjust_learning_rate(self.optimizer_A, i_iter, args.auxiliary)

            damping = (1 - i_iter/self.num_steps)

        # ======================================================================================
        # train G
        # ======================================================================================

            # Remove Grads in D
            for param in self.model_D.parameters():
                param.requires_grad = False

            # Train with Source
            _, batch = next(source_iter)
            images_s, labels_s, _, _ = batch
            images_s = images_s.to(device)
            pred_source1, pred_source2 = self.model(images_s)

            pred_source1 = interp_source(pred_source1)
            pred_source2 = interp_source(pred_source2)

        # Segmentation Loss
            loss_seg = (self.loss_calc(pred_source1, labels_s, device) + self.loss_calc(pred_source2, labels_s, device))
            loss_seg.backward()

        # Adversarial Loss
            if args.method.adversarial:
                # Train with Target
                _, batch = next(target_iter)
                images_t, _, _ = batch
                images_t = images_t.to(device)

                pred_target1, pred_target2 = self.model(images_t)

                pred_target1 = interp_target(pred_target1)
                pred_target2 = interp_target(pred_target2)

                weight_map = self.weightmap(F.softmax(pred_target1, dim=1), F.softmax(pred_target2, dim=1))

                D_out = interp_target(self.model_D(F.softmax(pred_target1 + pred_target2, dim=1)))

                # Adaptive Adversarial Loss
                if i_iter > self.preheat:
                    loss_adv = WeightedBCEWithLogitsLoss(D_out, torch.FloatTensor(D_out.data.size()).fill_(self.source_label).to(device),
                                                 weight_map, self.args.Epsilon, self.args.Lambda_local)
                else:
                    loss_adv = self.bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(self.source_label).to(device))

                loss_adv = loss_adv * self.args.Lambda_adv * damping
                loss_adv.backward()

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

                # Train with Target
                pred_target1 = pred_target1.detach()
                pred_target2 = pred_target2.detach()
                weight_map = weight_map.detach()

                D_out_t = interp_target(self.model_D(F.softmax(pred_target1 + pred_target2, dim=1)))

                # Adaptive Adversarial Loss
                if (i_iter > self.preheat):
                    loss_D_t = WeightedBCEWithLogitsLoss(D_out_t, torch.FloatTensor(D_out_t.data.size()).fill_(self.target_label).to(device),
                                                 weight_map, args.Epsilon, args.Lambda_local)
                else:
                    loss_D_t = self.bce_loss(D_out_t, torch.FloatTensor(D_out_t.data.size()).fill_(self.target_label).to(device))

                loss_D_t.backward()

        # ======================================================================================
        # Train SELF SUPERVISED TASK
        # ======================================================================================

            ''' SELF-SUPERVISED (ROTATION) ALGORITHM 
            - Get squared prediction --> resize
            - Rotate it randomly (0,90,180,270) -> assign self-label (0,1,2,3)  [*2 IF WANT TO CLASSIFY ALSO S/T]
            - Send rotated prediction to the classifier
            - Get loss 
            - Update weights of classifier and G (segmentation network) 
            '''

            if args.method.self:

                # Train with Source
                pred_source1 = pred_source1.detach()
                #pred_source2 = pred_source2.detach()

                # Train with Target
                pred_target1 = pred_target1.detach()
                #pred_target2 = pred_target2.detach()

                #save_prediction(pred_source1, './temp/before_square_color.png')
                pred_source = interp_prediction(pred_source1)
                pred_target = interp_prediction(pred_target1)

                #save_prediction(pred_source, './temp/square_color.png')

            #ROTATE TENSOR
                #source
                label_source = torch.empty(1, dtype=torch.long).random_(args.auxiliary.n_classes).to(device)
                rotated_pred_source = rotate_tensor(pred_source, self.rotations[label_source.item()])
                #save_prediction(pred_source, './temp/square_rotated_prediction_color.png')
                pred_source_label = self.model_A(rotated_pred_source)
                loss_rot_source = self.aux_loss(pred_source_label, label_source)

                #target
                #label_target = randint(0, args.auxiliary.n_classes - 1)
                label_target = torch.empty(1, dtype=torch.long).random_(args.auxiliary.n_classes).to(device)
                rotated_pred_target = rotate_tensor(pred_target, self.rotations[label_target.item()])
                pred_target_label = self.model_A(rotated_pred_target)
                loss_rot_target = self.aux_loss(pred_target_label, label_target)

                loss_rot = (loss_rot_source + loss_rot_target) * args.Lambda_aux

                loss_rot.backward()

            #Optimizers steps
            self.optimizer.step()
            if args.method.adversarial:
                self.optimizer_D.step()
            if args.method.self:
                self.optimizer_A.step()

            if i_iter % 10 == 0:
                print('Iter = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_adv = {3:.4f}, loss_weight = {4:.4f}, loss_D_s = {5:.4f} loss_D_t = {6:.4f}'.format(
                        i_iter, self.num_steps, loss_seg, loss_adv, loss_weight, loss_D_s, loss_D_t))
                print('Loss_rotation = {.4f}'.format(loss_rot))
            if (i_iter % args.save_pred_every == 0 and i_iter != 0) or i_iter == self.num_steps-1:
                print('saving weights...')
                torch.save(model.state_dict(), osp.join(snapshot_dir, 'GTA5_' + str(i_iter+1) + '.pth'))
                torch.save(model_D.state_dict(), osp.join(snapshot_dir, 'GTA5_' + str(i_iter+1) + '_D.pth'))

                #VALIDATE MODEL HERE...

        end = time.time()
        days = int((end - start) / 86400)
        print('Total training time: {} days, {} hours, {} min, {} sec '.format(
            days, int((end - start) / 3600)-(days*24), int((end - start) / 60 % 60),
            int((end - start) % 60)))
        print('### Experiment: ' + args.experiment + ' finished ###')


    def save(self, path, i_iter):
        state = {"iter": i_iter + 1,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                }
        save_path = os.path.join(path, 'model_{:06d}.pth'.format(i_iter))
        self.logger.info('Saving model to %s' % save_path)
        torch.save(state, save_path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.seg_model.load_state_dict(checkpoint['model_state'])
        self.logger.info('Loaded model from: ' + path)

        if self.args.mode == 'train':
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.start_iter = checkpoint['iter']
            self.logger.info('Start iter: %d ' % self.start_iter)

    def test(self, val_loader):
        val_loader_iterator = iter(val_loader)
        num_val_iters = len(val_loader)
        tt = tqdm(range(num_val_iters), total=num_val_iters, desc="Validating")

        aux_correct = 0
        class_correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for cur_it in tt:
                data = next(val_loader_iterator)
                if isinstance(data, list):
                    data = data[0]
                # Get the inputs
                data = to_device(data, self.device)
                imgs = data['images']
                cls_lbls = data['class_labels']
                aux_lbls = data['aux_labels']

                aux_logits, class_logits = self.model(imgs)

                _, cls_pred = class_logits.max(dim=1)
                _, aux_pred = aux_logits.max(dim=1)

                class_correct += torch.sum(cls_pred == cls_lbls.data)
                aux_correct += torch.sum(aux_pred == aux_lbls.data)
                total += imgs.size(0)

            tt.close()

        aux_acc = 100 * float(aux_correct) / total
        class_acc = 100 * float(class_correct) / total
        self.logger.info('aux acc: {:.2f} %, class_acc: {:.2f} %'.format(aux_acc, class_acc))
        return aux_acc, class_acc
