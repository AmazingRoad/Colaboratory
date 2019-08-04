import torch.nn as nn
import torch.nn.functional as F
import cv2

class LeNet5(nn.Module):

    def __init__(self, stage='train'):
        super(LeNet5, self).__init__()

        self.stage = stage
        self.conv1 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=5), nn.ReLU())
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(20, 50, kernel_size=3), nn.ReLU())
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(50, 80, kernel_size=3), nn.ReLU())
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(nn.Conv2d(80, 960, kernel_size=4), nn.ReLU())
        self.fc = nn.Linear(960, 43)
        
    def forward(self, x):

        x = self.conv1(x)
        if self.stage=='extract':
            save = x.data[0].cpu().numpy()
            for i, sub_map in enumerate(save):
                cv2.imwrite('./feature/conv1/{}.jpg'.format(i), sub_map)

        x = self.max_pool1(x)
        x = self.conv2(x)
        if self.stage=='extract':
            save = x.data[0].cpu().numpy()
            for i, sub_map in enumerate(save):
                cv2.imwrite('./feature/conv2/conv2_{}.jpg'.format(i), sub_map)

        x = self.max_pool2(x)
        x = self.conv3(x)
        if self.stage=='extract':
            save = x.data[0].cpu().numpy()
            for i, sub_map in enumerate(save):
                cv2.imwrite('./feature/conv3/{}.jpg'.format(i), sub_map)

        x = self.max_pool3(x)
        x = self.conv4(x)
        x = x.view(-1, 960)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output
