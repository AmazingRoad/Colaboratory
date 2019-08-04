import os
import argparse

parser = argparse.ArgumentParser(description='GTSRB dataset parser')
parser.add_argument('--root', type=str, default='./data/GTSRB_Final_Training_Images')
parser.add_argument('--save_path', type=str, default='./data/train.txt')

arg = parser.parse_args()

def parse_data(root, save):
    im_list = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if os.path.splitext(file)[-1] == '.ppm':
                im_list.append(os.path.join(root, file))

    with open(save, 'w') as save:
        for item in im_list:
            sub_item = item.split('/')
            label = int(sub_item[6]) 
            save.write('{} {}\n'.format(item, label))
    


if __name__ == '__main__':
    parse_data(arg.root, arg.save_path)