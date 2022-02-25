import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ConvNet_default(nn.Module):
    def __init__(self):
        super(ConvNet_default, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x


class ConvNet_featureMap(nn.Module):
    def __init__(self):
        super(ConvNet_featureMap, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5) # (color_channels, output_channel(arbitrary), kernel_size(5x5))
        self.pool = nn.MaxPool2d(2, 2) # (kernel_size, stride)
        self.conv2 = nn.Conv2d(64, 128, 5) # (input_channels, output_channel(arbitrary), kernel_size)
        
        self.fc1 = nn.Linear(128*5*5, 512) # (flatten vectors(num_channel*w*h),output_size(arbitrary))
        self.fc2 = nn.Linear(512, 256) # (input_features, output_size(arbitrary))
        self.fc3 = nn.Linear(256, 10) # (, num_class)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvNet_deeperLayers(nn.Module):
    def __init__(self):
        super(ConvNet_deeperLayers, self).__init__()
        # kernel means filter
        self.conv1 = nn.Conv2d(3, 16, 3) # (color_channels, output_channel(arbitrary), kernel_size(5x5))
        self.pool = nn.MaxPool2d(2, 2) # (kernel_size, stride)
        self.conv2 = nn.Conv2d(16, 32, 2) # (input_channels, output_channel(arbitrary), kernel_size)
        self.pool2 = nn.MaxPool2d(2, 2) # (kernel_size, stride)self.conv1 = nn.Conv2d(3, 16, 5) # (color_channels, output_channel(arbitrary), kernel_size(5x5))
        self.conv3 = nn.Conv2d(32, 64, 2) # (input_channels, output_channel(arbitrary), kernel_size)
        self.conv4 = nn.Conv2d(64, 128, 2) # (input_channels, output_channel(arbitrary), kernel_size)

        self.fc1 = nn.Linear(128 * 5 * 5, 256) # (flatten vectors(num_channel*w*h),output_size(arbitrary))
        self.fc2 = nn.Linear(256, 512) # (input_features, output_size(arbitrary))
        self.fc3 = nn.Linear(512, 10) # (, num_class)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 128 * 5 * 5)            # -> n, 400 # '-1' in order to let pytorch automatically find the correct size
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x


class ConvNet_alexnet(nn.Module):
    def __init__(self):
        super(ConvNet_alexnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool5= nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(256*1*1, 4096) # (flatten vectors(num_channel*w*h),output_size(arbitrary))
        self.fc2 = nn.Linear(4096, 4096) # (input_features, output_size(arbitrary))
        self.fc3 = nn.Linear(4096, 10) # (, num_class)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool1(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool2(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(-1, 256*1*1)            # -> n, 400 # '-1' in order to let pytorch automatically find the correct size
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x