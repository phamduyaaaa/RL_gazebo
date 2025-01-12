import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    def __init__(self):
        super(DuelingQNetwork, self).__init__()
        self.num_actions = 5
        self.num_quantiles = 51

        # Sửa số kênh đầu vào của conv1 thành 1
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=2)  # Chấp nhận 1 kênh đầu vào
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 32 * 40, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, self.num_actions * self.num_quantiles)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Tự động flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        quantiles = x.view(-1, self.num_actions, self.num_quantiles)
        return quantiles
