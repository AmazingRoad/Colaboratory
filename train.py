from core import LeNet5, GTSRBloader
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='LeNet-5 for GTSRB dataset')
parser.add_argument('--path', type=str, default='./data/train.txt', help='path to traning data text')
parser.add_argument('--dataset_name', type=str, default='GTSRB', help='dataset name')
parser.add_argument('--save_folder', default='weights/', help='Directory for saving checkpoint models')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')

parser.add_argument('--num_class', type=int, default=43, help='Total number of classes')
parser.add_argument('--im_size', type=int, default=48, help='Image size')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--max_epoch', default=100, type=int, help='Max epoch times')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,  help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def train():

    best_loss = 10000
    data_train = GTSRBloader(args.num_class, args.im_size, args.path, args.dataset_name)
    data_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    lenet = LeNet5()
    net = lenet

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.cuda:
        net = torch.nn.DataParallel(lenet)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        lenet.load_weights(args.resume)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        lenet.apply(weights_init)

    net.train()

    print('Loading the dataset...')

    print('Training on:', args.dataset_name)
    print('Using the specified args:')
    print(args)

    for epoch in range(args.max_epoch):
        print("[==> Training epoch %d]" % (epoch + 1))
        for i, (images, labels) in enumerate(data_loader):
            
            iteration = len(data_loader) * epoch + i

            if args.cuda:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)
            # forward
            
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss = F.nll_loss(out, labels)
            loss.backward()
            optimizer.step()
            
            iter_loss = loss.item()

            if iteration % 10 == 0:
                print('Training loss: {:.5f} in epoch: {}'.format(iter_loss, epoch))

            if best_loss > iter_loss:
                best_loss = iter_loss
                print('Saving state, iter:', iteration)
                torch.save(lenet.state_dict(), '{}/LeNet_GTSRB_best.pth'.format(args.save_folder))

            if epoch % 50 == 0 and (not epoch == 0):
                print('Saving state, iter:', iteration)
                torch.save(lenet.state_dict(), '{}/LeNet_GTSRB_epoch.pth'.format(args.save_folder))

def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

if __name__ == '__main__':
    train()
