from datetime import datetime
import os

def log_message(message, log_file):
    with open(log_file, 'a+') as f:
        f.write(message + '\n')
    print(message)


def init_log(log_file, args):
    if not (os.path.exists(log_file)):
        with open(log_file, 'w+') as f:
            time = datetime.now().strftime('%d/%m/%Y - %H:%M:%S')
            f.write('time: ' + time + '\n\n')

## TODO: ADD MODEL HYPERPARAMETERS ON FILE