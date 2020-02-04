from datetime import datetime
import logging
import os
import sys


def log_message(message, log_file):
    with open(log_file, 'a+') as f:
        f.write(message + '\n')
    print(message)


def init_log(log_file):
    if not (os.path.exists(log_file)):
        with open(log_file, 'w+') as f:
            time = datetime.now().strftime('%d/%m/%Y - %H:%M:%S')
            f.write('time: ' + time + '\n\n')

def get_logger(logdir, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")

    file_path = os.path.join(logdir, "log.txt")
    open(file_path, 'w+')
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)

    logger.info('DateTime: ' + datetime.now().strftime('%d/%m/%Y - %H:%M:%S'))
    return logger
