import os
from os.path import join


def create_dirs(dirs, exp_name):
    try:
        for dir_ in dirs:
            new_dir = join(dir_, exp_name)
            os.makedirs(new_dir, exist_ok=True)
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
