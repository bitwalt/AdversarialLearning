import json
import yaml
from easydict import EasyDict
import os
from os.path import join
from shutil import copyfile
import argparse

CONFIG = './config/gta_to_cityscapes.yaml'


def get_config_from_json(json_file):
    """
    Get the config from a json file
    Input:
        - json_file: json configuration file
    Return:
        - config: namespace
        - config_dict: dictionary
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config, config_dict


def get_config_from_yaml(yaml_file):
    """
    Get the config from yaml file
    Input:
        - yaml_file: yaml configuration file
    Return:
        - config: namespace
        - config_dict: dictionary
    """

    with open(yaml_file) as fp:
        config_dict = yaml.load(fp, Loader=yaml.FullLoader)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)
    return config, config_dict


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-c', '--config', metavar='C', default=CONFIG, help='The Configuration file')
    argparser.add_argument('-e', type=str, help='Experiment name')
    args = argparser.parse_args()
    return args


def process_config(config_file):
    if config_file.endswith('json'):
        config, _ = get_config_from_json(config_file)
    elif config_file.endswith('yaml'):
        config, _ = get_config_from_yaml(config_file)
    else:
        raise Exception("Only .json and .yaml are supported!")

    create_dirs([config.dirs.snapshot_dir, config.dirs.prediction_dir, config.dirs.output_dir], config.experiment)

    config.snapshot_dir = join(config.dirs.snapshot_dir, config.experiment)
    config.prediction_dir = join(config.dirs.prediction_dir, config.experiment)
    config.output_dir = join(config.dirs.output_dir, config.experiment)

    config.results_dir = join(config.output_dir, 'results')
    config.log_dir = join(config.output_dir, 'log')
    config.miou_dir = join(config.results_dir, 'mIoUs')
    config.images_dir = join(config.results_dir, 'images')
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.miou_dir, exist_ok=True)
    os.makedirs(config.images_dir, exist_ok=True)

    copyfile(config_file, join(config.log_dir, 'config.yaml'))

    return config


def create_dirs(dirs, exp_name):
    try:
        for dir_ in dirs:
            new_dir = join(dir_, exp_name)
            os.makedirs(new_dir, exist_ok=True)
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

