import torch
import numpy as np
from PIL import Image
import os
from os.path import join

import torchvision.transforms.functional as F

from torchvision.utils import save_image

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


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


def save_prediction(tensor, file):
    output = tensor.cpu().data[0].numpy()
    output = output.transpose(1, 2, 0)
    output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
    output_col = colorize_mask(output)
    # output = Image.fromarray(output)
    output_col.save(file)


def save_rotations(save_dir, pred, rotated, iter):
    save_path = join(save_dir, 'orig.png')
    save_prediction(pred, save_path)

    save_path = join(save_dir, '90.png')
    p = rotate_tensor(pred, 90)
    save_prediction(p, save_path)

    save_path = join(save_dir, '180.png')
    p = rotate_tensor(pred, 180)
    save_prediction(p, save_path)

    save_path = join(save_dir, '270.png')
    p = rotate_tensor(pred, 270)
    save_prediction(p, save_path)
    #save_path = join(save_dir, 'pred_%s_rotated.png' % iter)
    #save_prediction(rotated, save_path)


def save_segmentations(save_dir, source_im, source_gt, source_pred, target_im):
    # Save: source im, source gt, source pred
    #       target im, target gt, target pred

    save_path = join(save_dir, 'source.png')
#    save_images(F.to_pil_image(source_im[0].cpu().numpy()), save_path)

    save_images(tensor2im(source_im ), save_path)
    save_image(source_im[0], save_path, normalize=True)

    save_path = join(save_dir, 'target.png')
    save_images(tensor2im(target_im), save_path)

    save_path = join(save_dir, 'source_label.png')
#    save_prediction(source_gt, save_path)

    save_path = join(save_dir, 'source_predict.png')
    save_prediction(source_pred, save_path)


def tensor2im(image_tensor, imtype=np.uint8):
    #print(image_tensor.size())
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #image_numpy = (std * image_numpy + mean) * 255
    #image_numpy = (image_numpy + IMG_MEAN) * 255
    return image_numpy.astype(imtype)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def decode_segmap(seg_pred):
    """
    Create RGB images from segmentation predictions.
    """
    pred = seg_pred.data.cpu().numpy()
    pred = np.argmax(pred, axis=0)
    img = mask2color(pred)
    img = np.transpose(img, (2, 0, 1)) # h, w, c -> c, h, w
    return img


def weightmap(pred1, pred2):
    output = 1.0 - torch.sum((pred1 * pred2), 1).view(1, 1, pred1.size(2), pred1.size(3)) / \
             (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(1, 1, pred1.size(2), pred1.size(3))
    return output


def save_images(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
