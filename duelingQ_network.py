import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, vgg11

class DuelingQNetwork(nn.Module):
    def __init__(self):
        super(DuelingQNetwork, self).__init__()
        
        self.num_actions = 5
        self.num_quantiles = 51

        # ResNet-18 (chỉ lấy phần convolutional)
        self.resnet = resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Loại bỏ lớp fully connected
        self.resnet_output_size = 512  # ResNet-18 có đầu ra 512 đặc trưng

        # VGG-11 (chỉ lấy phần convolutional)
        self.vgg = vgg11(pretrained=True)
        self.vgg = nn.Sequential(*list(self.vgg.features),  # Lấy phần feature layers
                                 nn.AdaptiveAvgPool2d((1, 1)))  # Pooling để cố định kích thước đầu ra
        self.vgg_output_size = 512  # VGG-11 có đầu ra 512 đặc trưng

        # Fully connected layers để kết hợp đặc trưng
        self.fc1 = nn.Linear(self.resnet_output_size + self.vgg_output_size, 512)
        self.fc_adv = nn.Linear(512, self.num_actions * self.num_quantiles)  # Advantage stream
        self.fc_val = nn.Linear(512, self.num_quantiles)  # Value stream

    def forward(self, x):
        # Input x: (batch_size, 1, height, width), phải mở rộng thành (batch_size, 3, height, width)
        x = x.repeat(1, 3, 1, 1)

        # Tính đặc trưng từ ResNet-18
        resnet_features = self.resnet(x)
        resnet_features = torch.flatten(resnet_features, 1)

        # Tính đặc trưng từ VGG-11
        vgg_features = self.vgg(x)
        vgg_features = torch.flatten(vgg_features, 1)

        # Kết hợp đặc trưng
        combined_features = torch.cat((resnet_features, vgg_features), dim=1)
        x = F.relu(self.fc1(combined_features))

        # Advantage và Value streams
        adv = self.fc_adv(x).view(-1, self.num_actions, self.num_quantiles)
        val = self.fc_val(x).view(-1, 1, self.num_quantiles)

        # Tính Q-values theo cách Dueling
        quantiles = val + adv - adv.mean(dim=1, keepdim=True)
        
        return quantiles
