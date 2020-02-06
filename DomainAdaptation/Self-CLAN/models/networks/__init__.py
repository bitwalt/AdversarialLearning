from models.networks.caffenet import caffenet
from models.networks.mnist import lenet
from models.networks.resnet import resnet18, resnet50
from models.networks.alexnet import alexnet

nets_map = {
    'caffenet': caffenet,
    'alexnet': alexnet,
    'resnet18': resnet18,
    'resnet50': resnet50,
    'lenet': lenet
}


def get_auxiliary_net(name):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_network_fn
