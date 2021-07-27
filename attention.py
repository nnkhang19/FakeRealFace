import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import spectral_norm 


# reference: Self Attention Generative Adversarial Networks  

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.block = spectral_norm(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        return self.block(x)



class SelfAttention(nn.Module):
    def __init__(self, in_channels, k = 8):
        super(SelfAttention, self).__init__()

        self.f = Conv1x1(in_channels, in_channels // k)
        self.g = Conv1x1(in_channels, in_channels // k)
        self.h = Conv1x1(in_channels, in_channels // k)
        self.v = Conv1x1(in_channels // k, in_channels)

        self.softmax = nn.Softmax(dim= -1)
        self.llambda = nn.Parameter(torch.zeros(1))


    def forward(self, x, y):

        f_x = self.f(x)
        g_x = self.g(y)
        h_x = self.h(x)


        # Dot product between f_x and g_x : (N, C, W, H).T * (N, C, W, H) -> (N, W, H)
        # C = in_channels / k

        attend = torch.einsum('ncwh, ncwh -> nwh', [f_x, g_x])
        attend = self.softmax(attend)

        
        # Dot product between h_x and atten: (N, C, W, H) * (N, W, H) -> (N, C, W, H) 
        # C = in_channels / k
        theta = torch.einsum('ncwh, nwh -> ncwh', [h_x, attend])
        theta = self.v(theta) 

        output = x + self.llambda * theta 

        return output 