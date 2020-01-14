from datetime import datetime


def log_message(message, log_file):
    with open(log_file, 'a') as f:
        f.write(message + '\n')
    print(message)


def init_log(log_file):
    with open(log_file, 'a') as f:
        time = datetime.now().strftime('%d/%m/%Y - %H:%M:%S')
        f.write('time: ' + time + '\n')