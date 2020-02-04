import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os, random
from os.path import join
import time, timeit, datetime
from models.CLAN_G import Res_Deeplab

from utils.log import log_message, init_log
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

from utils.config import process_config, get_args


from utils.optimizer import adjust_learning_rate
from utils.loss import WeightedBCEWithLogitsLoss, loss_calc
from utils.loss import save_losses_plot
from utils.utils import rotate_tensor
from utils.visual import colorize_mask
from iou import compute_mIoU
from dataset.data_loader import get_source_train_dataloader, get_target_val_dataloader
from dataset.data_loader import get_target_train_dataloader
from utils.log import get_logger

from networks import get_auxiliary_net

def main(args):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    losses = {'seg': list(), 'adv': list(), 'weight': list(), 'ds': list(), 'dt': list(), 'aux': list()}

    cudnn.enabled = True
    cudnn.benchmark = True

    # logging to the file and stdout
    log = get_logger(args.log_dir, args.experiment)

    # fix random seed to reproduce results
    random.seed(args.random_seed)
    log.info('Random seed: {:d}'.format(args.random_seed))


    if args.model.name == 'DeepLab':
        model = Res_Deeplab(num_classes=args.num_classes, restore_from=args.model.restore_from)
    model.train()
    model.to(device)


    model_A = get_auxiliary_net(args.auxiliary.name)(input_dim=args.auxiliary.classes, aux_classes=args.auxiliary.n_classes, classes=args.auxiliary.classes, pretrained=False)
    model_A.train()
    model_A = model_A.to(device)
    optimizer_A = optim.Adam(model_A.parameters(), lr=args.discriminator.optimizer.lr, betas=(0.9, 0.99))
    optimizer_A.zero_grad()

# DataLoaders
    source_loader = get_source_train_dataloader(args.datasets.source)
    target_loader = get_target_train_dataloader(args.datasets.target)
    #val_loader = get_target_val_dataloader(args.datasets.target)
    source_iter = enumerate(source_loader)
    target_iter = enumerate(target_loader)

    # Optimizers
    optimizer = optim.SGD(model.optim_parameters(args.model.optimizer), lr=args.model.optimizer.lr,
                               momentum=args.model.optimizer.momentum, weight_decay=args.model.optimizer.weight_decay)

    optimizer.zero_grad()


    # Losses
    bce_loss = torch.nn.BCEWithLogitsLoss()
    weighted_bce_loss = WeightedBCEWithLogitsLoss()
    aux_loss = nn.CrossEntropyLoss()

    rotations = [0, 90, 180, 270]
    # Labels for Adversarial Training
    source_label = 0
    target_label = 1
    loss_rot = loss_adv = loss_weight = loss_D_s = loss_D_t = 0

    preheat = args.num_steps / 20
    # ======================================================================================
    # Start training
    # ======================================================================================
    print('###########   TRAINING STARTED  ############')

    interp_source = nn.Upsample(size=(args.datasets.source.images_size[1], args.datasets.source.images_size[0]), mode='bilinear', align_corners=True)
    interp_prediction = nn.Upsample(size=(args.auxiliary.images_size[1], args.auxiliary.images_size[0]), mode='bilinear', align_corners=True)

    start = time.time()

    for i_iter in range(args.start_iter, args.num_steps):

        model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, preheat, args.num_steps, args.power, i_iter, args.model.optimizer)

        model_A.train()
        optimizer_A.zero_grad()
        adjust_learning_rate(optimizer_A, preheat, args.num_steps, args.power, i_iter, args.auxiliary.optimizer)


        damping = (1 - i_iter / args.num_steps)  # similar to early stopping

        # ======================================================================================
        # train G
        # ======================================================================================

        # Train with Source
        _, batch = next(source_iter)
        images_s, labels_s, _, _ = batch
        images_s = images_s.to(device)
        pred_source1_, pred_source2_ = model(images_s)

        pred_source1 = interp_source(pred_source1_)
        pred_source2 = interp_source(pred_source2_)

        # Segmentation Loss
        loss_seg = (loss_calc(args.num_classes, pred_source1, labels_s, device) + loss_calc(args.num_classes, pred_source2, labels_s, device))
        loss_seg.backward()
        losses['seg'].append(loss_seg.item())


        _, batch = next(target_iter)
        images_t, _ = batch  # get target label here
        images_t = images_t.to(device)
        pred_target1_, pred_target2_ = model(images_t)
        # TODO: ADD SEGMENTATION LOSS ALSO X TARGET -> semi-supervised approach


        # Weight Discrepancy Loss
        if args.weight_loss:
            W5 = None
            W6 = None
            if args.model.name == 'DeepLab':

                for (w5, w6) in zip(model.layer5.parameters(), model.layer6.parameters()):
                    if W5 is None and W6 is None:
                        W5 = w5.view(-1)
                        W6 = w6.view(-1)
                    else:
                        W5 = torch.cat((W5, w5.view(-1)), 0)
                        W6 = torch.cat((W6, w6.view(-1)), 0)

            loss_weight = (torch.matmul(W5, W6) / (torch.norm(W5) * torch.norm(W6)) + 1)  # +1 is for a positive loss
            loss_weight = loss_weight * args.Lambda_weight * damping * 2
            loss_weight.backward()
            losses['weight'].append(loss_weight.item())


        # ======================================================================================
        # Train SUPERVISED TASK
        # ======================================================================================


            ''' SUPERVISED (ROTATION) ALGORITHM 
            - Get squared prediction 
            - Rotate it randomly (0,90,180,270) -> assign label (0,1,2,3)  [*2 IF WANT TO CLASSIFY ALSO S/T]
            - Send rotated prediction to the classifier
            - Get loss 
            - Update weights of classifier and G (segmentation network) 
            '''

            # TODO: GET RANDOM PREDICTION
            # Train with Source

        pred_source1_ = pred_source1_.detach()
        pred_source2_ = pred_source2_.detach()

            # Train with Target
            #pred_target1 = pred_target1_.detach()
            #pred_target2 = pred_target2_.detach()

            # save_prediction(pred_source1, './temp/before_square_color.png')
            # pred_s =
        pred_source = interp_prediction(F.softmax(pred_source1_ + pred_source2_, dim=1))
        #pred_target = interp_prediction(F.softmax(pred_target1_ + pred_target2_, dim=1))

        # save_prediction(pred_source, './temp/square_color.png')

        # ROTATE TENSOR
        # source
        label_source = torch.empty(1, dtype=torch.long).random_(args.auxiliary.n_classes).to(device)
        rotated_pred_source = rotate_tensor(pred_source, rotations[label_source.item()])
        # target
        #label_target = torch.empty(1, dtype=torch.long).random_(args.auxiliary.n_classes).to(device)
        #rotated_pred_target = rotate_tensor(pred_target, rotations[label_target.item()])


        pred_source_label = model_A(rotated_pred_source)
        #pred_target_label = model_A(rotated_pred_target)

        loss_rot_source = aux_loss(pred_source_label, label_source)
        #loss_rot_target = aux_loss(pred_target_label, label_target)

        #loss_rot = (loss_rot_source + loss_rot_target) * args.Lambda_aux

        loss_rot_source.backward()
        losses['aux'].append(loss_rot_source.item())

        del pred_source1,pred_source2_,pred_source

        # Optimizers steps
        optimizer.step()
        optimizer_A.step()

        if i_iter % 10 == 0:
            log.info('Iter = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_rot = {3:.4f}, loss_adv = {4:.4f}, loss_weight = {5:.4f}, loss_D_s = {6:.4f} loss_D_t = {7:.4f}'.format(i_iter, args.num_steps, loss_seg, loss_rot, loss_adv, loss_weight, loss_D_s, loss_D_t))

        if (i_iter % args.save_pred_every == 0 and i_iter != 0) or i_iter == args.num_steps - 1:
            log.info('saving weights...')
            i_iter = i_iter if i_iter != args.num_steps - 1 else i_iter + 1  # for last iter
            #torch.save(model.state_dict(), join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            #if args.method.adversarial:
                #torch.save(model_D.state_dict(), join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pth'))
                # SAVE MAPS

            #if args.method.self:
                #torch.save(model_A.state_dict(), join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_Aux.pth'))
                #SAVE ROTATION

            #validate(args.prediction_dir, model, i_iter, val_loader, device)
            #compute_mIoU(i_iter, args.datasets.target.val.label_dir, save_path, args.datasets.target.json_file, args.datasets.target.base_list, args.results_dir)

            # SAVE ALSO IMAGES OF SOURCE AND TARGET
            #save_segmentations(args.images_dir, images_s, labels_s  )

            save_losses_plot(args.results_dir, losses)


    end = time.time()
    days = int((end - start) / 86400)
    log.info('Total training time: {} days, {} hours, {} min, {} sec '.format(days, int((end - start) / 3600) - (days * 24), int((end - start) / 60 % 60), int((end - start) % 60)))
    print('### Experiment: ' + args.experiment + ' finished ###')


def validate(prediction_dir, model, current_iter, val_loader, device):
    model.eval()
    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
    save_path = join(prediction_dir, str(current_iter))
    os.makedirs(save_path, exist_ok=True)

    print('### STARTING EVALUATING ###')
    print('total to process: %d' % len(val_loader))
    with torch.no_grad():
        for index, batch in enumerate(val_loader):
            if index % 100 == 0:
                print('%d processed' % index)
            image, _, name, = batch
            output1, output2 = model(image.to(device))
            output = interp(output1 + output2).cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            output_col = colorize_mask(output)
            output = Image.fromarray(output)

            name = name[0].split('/')[-1]
            output.save('%s/%s' % (save_path, name))
            output_col.save('%s/%s_color.png' % (save_path, name.split('.')[0]))

    print('### EVALUATING FINISHED ###')


if __name__ == '__main__':
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system('rm tmp')
    gpu_target = str(np.argmax(memory_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_target
    print('Training on GPU ' + gpu_target)

    args = get_args()
    args = process_config(args.config)
    main(args)
