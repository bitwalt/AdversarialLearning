import os
import os.path as osp
import numpy as np
import random
#import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
import json
from collections import namedtuple
from PIL import Image

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

class cityscapesDataSetLabel(data.Dataset):

    def __init__(self, root, json_file, list_path, label_list, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), set='val', label_folder='gtFine'):
        self.root = root
        self.list_path = list_path
        self.label_list = label_list
        self.crop_size = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.label_ids = [i_id.strip() for i_id in open(label_list)]

        with open(json_file, 'r') as fp:
            info = json.load(fp)
        self.mapping = np.array(info['label2train'], dtype=np.int)

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))

        self.files = []
        self.set = set
        self.label_folder = label_folder

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        l = self.label_ids[index]
        image = Image.open(osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))).convert('RGB')
        label = Image.open(osp.join(self.root, "gtFine/%s/%s" % (self.set, l)))

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.array(image, np.float32)
        label = np.array(label, np.float32)

        label = label_mapping(label, self.mapping)
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        return image.copy(), label.copy()
