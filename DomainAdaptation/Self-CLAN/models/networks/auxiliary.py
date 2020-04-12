import torch
import torch.nn as nn
from models.networks.caffenet import caffenet
from models.networks.mnist import lenet
from models.networks.resnet import resnet18, resnet50
from models.networks.alexnet import alexnet


def auxiliary(name='cnn', input_dim=19, aux_classes=4, restore_from=None):

    model_A = None
    if name == 'cnn':
        model_A = AuxiliaryNet(input_dim=input_dim, aux_classes=aux_classes)
    elif name == 'resnet18':
        model_A = resnet18(input_dim=input_dim, aux_classes=aux_classes)
    elif name == 'resnet50':
        model_A = resnet50(input_dim=input_dim, aux_classes=aux_classes)
    elif name == 'alexnet':
        model_A = alexnet(input_dim=input_dim, aux_classes=aux_classes)
    else:
        print('Network model not defined')
        exit(0)

    if restore_from != 'None':
        saved_state_dict = torch.load(restore_from)
        model_A.load_state_dict(saved_state_dict)

    return model_A


class Id(nn.Module):
    def __init__(self):
        super(Id, self).__init__()

    def forward(self, x):
        return x


class AuxiliaryNet(nn.Module):
    def __init__(self, input_dim, aux_classes, dropout=True, ndf=64):
        super(AuxiliaryNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_dim, ndf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout() if dropout else Id(),
            nn.Linear(4608, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout() if dropout else Id(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, aux_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4608)
        x = self.classifier(x)
        return x
