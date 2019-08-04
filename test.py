from core import LeNet5, GTSRBloader
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='LeNet-5 for GTSRB dataset')
parser.add_argument('--im_path', type=str, default='./data/feature.jpg', help='path to image')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--im_size', type=int, default=48, help='Image size')
parser.add_argument('--model_path', default='weights/LeNet_GTSRB_best.pth', type=str, help='Trained model path')

args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def test():

    net = LeNet5('extract')
    net.load_state_dict(torch.load(args.model_path))
    net.eval()

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    img = cv2.imread(args.im_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (args.im_size, args.im_size))
    img = img[:,:, np.newaxis]
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = Variable(img.unsqueeze(0))

    if args.cuda:
        img = img.cuda()
    
    out = net(img)
    predcition = out.data
    index = np.argmax(predcition)
    print('Predict label: {}'.format(index))

        
if __name__ == '__main__':
    test()
