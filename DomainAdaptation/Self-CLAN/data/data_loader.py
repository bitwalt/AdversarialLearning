import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

def get_source_train_dataloader(args):

    source_loader = DataLoader(
        GTA5DataSet(args.source_dir, args.source_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=args.images_size, scale=True, mirror=True, mean=args.mean),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    #trainloader_iter = enumerate(trainloader)
    return source_loader


def get_target_train_dataloader(args):

    target_loader = DataLoader(
        cityscapesDataSet(args.target_dir, args.target_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                          crop_size=args.images_size, scale=True, mirror=True, mean=args.mean, set=args.set),
                          batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    #targetloader_iter = enumerate(targetloader)
    return target_loader


def get_target_val_dataloader(args):

    val_loader = DataLoader(
        cityscapesDataSet(args.target_dir, args.val.list, crop_size=args.val.images_size, mean=args.mean, scale=False,
                          mirror=False, set=args.val.set), batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return val_loader