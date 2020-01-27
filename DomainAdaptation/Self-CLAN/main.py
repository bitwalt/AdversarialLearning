import random, os
import numpy as np
from utils.config import process_config, get_args
from utils.log import get_logger

from data.data_loader import get_source_train_dataloader
from data.data_loader import get_target_train_dataloader
from data.data_loader import get_target_val_dataloader
from Self_CLAN import Self_CLAN


def main():
    args = get_args()
    config = process_config(args.config)

    # logging to the file and stdout
    logger = get_logger(config.log_dir, config.experiment)

    # fix random seed to reproduce results
    random.seed(config.random_seed)
    logger.info('Random seed: {:d}'.format(config.random_seed))

    model = Self_CLAN(config, logger)
    
    # Get train dataloader
    source_loader = get_source_train_dataloader(config.datasets.source)
    target_loader = get_target_train_dataloader(config.datasets.target)

    # Get validation dataloader
    val_loader = get_target_val_dataloader(config.datasets.target)

    if config.mode == 'train':
        model.train(source_loader, target_loader, val_loader)

    #elif config.mode == 'test': # NOT IMPLEMENTED - da capire se bastano risultati sul validation set
    #    model.test(test_loader)

if __name__ == '__main__':
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system('rm tmp')
    gpu_target = str(np.argmax(memory_gpu))
    #gpu_target = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_target
    print('Training on GPU ' + gpu_target)
    main()
