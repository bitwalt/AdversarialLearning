from PIL import Image
import os

def convertRGBtoGrey(in_dir, out_dir):

    print('Start converting images to grey scale...')
    os.makedirs(out_dir, exist_ok=True)
    i=0
    for file in os.listdir(in_dir):
        i+=1
        img = Image.open(in_dir+file).convert('LA')
        img.save(out_dir+file)
        if i%100==0:
            print(str(i))
    print('## Images converted ##')

in_dir = '/Users/walter/Desktop/ML/datasets/GTA_V/images/'
out_dir = '/Users/walter/Desktop/ML/datasets/GTA_V/grey_images/'
convertRGBtoGrey(in_dir, out_dir)