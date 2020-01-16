import os
from random import seed
from random import randint
# seed random number generator
seed(10)

def create_list_gta(dir,n):
    file = os.path.join(dir, 'train%d.txt' % n)
    with open(file , 'w+') as fp:
        for _ in range(n):
            value = randint(2, 24966)
            fp.write('%05d' % value + '.png\n')
    print('file written')

if __name__ == '__main__':
    dir = './dataset/gta5_list/'
    n = 10000
    create_list_gta(dir, n)