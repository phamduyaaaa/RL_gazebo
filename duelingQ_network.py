import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    def __init__(self):
        super(DuelingQNetwork, self).__init__()
        self.num_actions = 5
        self.num_quantiles = 51
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        
        
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 5 * 51)

    def forward(self, x):
   
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        quantiles = x.view(-1, self.num_actions, self.num_quantiles)
        
        return quantiles
