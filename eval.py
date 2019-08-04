from core import LeNet5, GTSRBloader
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='LeNet-5 for GTSRB dataset')
parser.add_argument('--path', type=str, default='./data/train.txt', help='path to traning data text')
parser.add_argument('--dataset_name', type=str, default='GTSRB', help='dataset name')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')

parser.add_argument('--num_class', type=int, default=43, help='Total number of classes')
parser.add_argument('--im_size', type=int, default=48, help='Image size')
parser.add_argument('--model_path', default='weights/LeNet_GTSRB_best.pth', type=str, help='Trained model path')

args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def test():

    data_test = GTSRBloader(args.num_class, args.im_size, args.path, args.dataset_name)

    num_images = len(data_test)
    correct = 0

    net = LeNet5()
    net.load_state_dict(torch.load(args.model_path))
    net.eval()

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    for i in range(num_images):

        print('Process {} / {} images...'.format(i+1, num_images))

        img, label = data_test.pull_image(i)
        img = Variable(img.unsqueeze(0))

        if args.cuda:
            img = img.cuda()
        
        out = net(img)
        predcition = out.data
        index = np.argmax(predcition)
        if(index == label):
            correct += 1

    print('Accuray: {}%'.format(float(correct) / num_images * 100))
        
if __name__ == '__main__':
    test()
