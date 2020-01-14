import os

dir = '/media/data/walteraul_data/datasets/gta5/images'
files = 'dataset/gta5_list/train.txt'


i=0
with open(files) as fp:
    for line in fp:
            i+=1


print('tot: ', i)

present = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
print('present: ', present)

print('missing: ' , i-present)



