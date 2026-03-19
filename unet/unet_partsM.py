import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import LeakyReLU

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True)
            
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(out_channels, affine=False),
            self.activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(out_channels, affine=False),
            self.activation
        )
        
    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu'):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, activation=activation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        d = self.conv(x)
        p = self.pool(d)

        return d, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu'):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, activation=activation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        g1 = F.interpolate(g1, size=x.size()[2:], mode="bilinear", align_corners=True)
        x1 = self.W_x(x)
        alpha = self.psi(g1 + x1)
        return x * alpha

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += self.shortcut(residual) 
        out = self.relu(out)
        return out

class RecurrentConv(nn.Module):
    def __init__(self, out_channels, t=2):
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.conv(x)
        for _ in range(self.t - 1):
            x1 = self.conv(x + x1)
        return x1
    
class RecConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.rc_layer = RecurrentConv(out_channels, t=t)

    def forward(self, x):
        x = self.conv(x)
        x = self.rc_layer(x)
        return x
    
class R2ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super().__init__()
        # transforms input to output channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.rc_layer = RecurrentConv(out_channels, t=t)

    def forward(self, x):
        residual = self.conv(x)
        x = self.rc_layer(residual)
        return x + residual