import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    def __init__(self, num_actions=5, num_quantiles=51):
        super(DuelingQNetwork, self).__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers for feature extraction
        self.fc1 = nn.Linear(64 * 32 * 40, 512)
        self.dropout = nn.Dropout(p=0.5)

        # Dueling branches
        self.value_fc = nn.Linear(512, self.num_quantiles)  # Nhánh Value
        self.advantage_fc = nn.Linear(512, self.num_actions * self.num_quantiles)  # Nhánh Advantage

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Dueling branches
        value = self.value_fc(x)  # Value nhánh
        advantage = self.advantage_fc(x)  # Advantage nhánh

        # Reshape để tính toán
        value = value.view(-1, 1, self.num_quantiles)  # (batch_size, 1, num_quantiles)
        advantage = advantage.view(-1, self.num_actions, self.num_quantiles)  # (batch_size, num_actions, num_quantiles)

        # Combine Value và Advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))  # Áp dụng công thức

        return q_values
