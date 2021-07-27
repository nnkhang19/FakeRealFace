import torch
import torch.nn as nn 
from torchvision import models
import numpy as np 
from blocks import ResidualBlock, DownSamplingBlock, Block
from attention import SelfAttention

class AttentionDiscriminator(nn.Module):
    def __init__(self, image_channels = 3, features = [64, 128, 256, 512]):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(image_channels, features[0], kernel_size=4, stride=2, padding = 1, padding_mode="reflect"), 
            nn.LeakyReLU(0.2)
        )

        self.layers = nn.ModuleList()
        self.attends = nn.ModuleList()

        in_channels = features[0]

        for feature in features[1:]:
            block = DownSamplingBlock(in_channels, feature, stride = 1 if feature == features[-1] else 2)
            attend = SelfAttention(feature)
            self.layers.append(block)
            self.attends.append(attend)
            in_channels = feature


        #mapping = nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        self.mapping = nn.Conv2d(in_channels, 1, 4, 1, 1)
        self.linear = nn.Linear(900, 1)

    def forward(self, x):
        x = self.initial(x)
        for module, attend in zip(self.layers, self.attends):
            x = module(x)
            x = attend(x, x)
        
        x = self.mapping(x)
        #print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.linear(x).squeeze()
        
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, features =[64,128,256,512]):
        super(Discriminator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding = 1, padding_mode="reflect"), 
            nn.LeakyReLU(0.2)
        )

        self.layers = nn.ModuleList()

        in_channels = features[0]

        for feature in features[1:]:
            block = DownSamplingBlock(in_channels, feature, stride = 1 if feature == features[-1] else 2)
            self.layers.append(block)
            in_channels = feature


        #mapping = nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        mapping = nn.Conv2d(in_channels, 1, 4, 1, 1)
        self.layers.append(mapping)

        self.linear = nn.Linear(900, 1)

    def forward(self, x):
        x = self.initial(x)
        for module in self.layers:
            x = module(x)

        x = x.view(x.shape[0], -1)
        x = self.linear(x).squeeze()
        
        return x
    
