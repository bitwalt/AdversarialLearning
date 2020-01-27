import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data, model_zoo
from models.CLAN_G import Res_Deeplab
from dataset.cityscapes_dataset import cityscapesDataSet
import os
from PIL import Image
import torch.nn as nn

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

SAVE_PATH = '/media/data/walteraul_data/results/10k_5000GTA/checkpoint10000' # save prediction here
RESTORE_FROM = './snapshots/10k_5000GTA/GTA5_10000.pth'          # Restore from this checkpoint

DATA_DIRECTORY = '/media/data/walteraul_data/datasets/cityscapes'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'


IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
SET = 'val'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("-data-dir", type=str, default=DATA_DIRECTORY, help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("-data-list", type=str, default=DATA_LIST_PATH, help="Path to the file listing the images in the dataset.")
    parser.add_argument("-ignore-label", type=int, default=IGNORE_LABEL, help="The index of the label to ignore during the training.")
    parser.add_argument("-num-classes", type=int, default=NUM_CLASSES, help="Number of classes to predict (including background).")
    parser.add_argument("-fromm", type=str, default=RESTORE_FROM, help="Where restore model parameters from.")
    parser.add_argument("-cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("-gpu", type=int, default=0, help="choose gpu device.")
    parser.add_argument("-set", type=str, default=SET, help="choose evaluation set.")
    parser.add_argument("-save", type=str, default=SAVE_PATH, help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    os.makedirs(args.save, exist_ok=True)

    device = torch.device("cuda" if not args.cpu else "cpu")

    model = Res_Deeplab(num_classes=args.num_classes)

    saved_state_dict = torch.load(args.fromm)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.to(device)

    testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

    print('### STARTING EVALUATING ###')
    print('total to process: %d' % len(testloader))
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            if index % 100 == 0:
                print('%d processed' % index)
            image, _, name, = batch
            output1, output2 = model(image.to(device))

            output = interp(output1 + output2).cpu().data[0].numpy()
            
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
    
            output_col = colorize_mask(output)
            output = Image.fromarray(output)
    
            name = name[0].split('/')[-1]
            output.save('%s/%s' % (args.save, name))

            output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))

    print('### EVALUATING FINISHED ###')

if __name__ == '__main__':
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system('rm tmp')
    gpu_target = str(np.argmax(memory_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_target
    print('Training on GPU ' + gpu_target)
    main()
