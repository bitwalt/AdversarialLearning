import os


list_path = './train.txt'
file_name = './train_label.txt'

if __name__ == '__main__':
    img_ids = [i_id.strip() for i_id in open(list_path)]

    with open(file_name, 'w+') as f:
        for im_id in img_ids:
            s = im_id.split('_', 3)
            string = s[0]+'_' + s[1] + '_' + s[2] + '_gtFine_labelIds.png'
            f.write(string + '\n')
    print('file written')
