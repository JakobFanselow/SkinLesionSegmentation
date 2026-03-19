import torch
import torch.nn as nn


class InputConvolution(nn.Module):
    def __init__(self, in_channels=3,out_channels=16,kernel_size=5):
        super(InputConvolution, self).__init__()
        padding = int((kernel_size-1)/2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        out = self.conv(x) 
        return self.prelu(out)

class FeatureFusionBlock(nn.Module):
    def __init__(self, in_channels_sum,out_channels,kernel_size=5):
        super(FeatureFusionBlock,self).__init__()
        padding = int((kernel_size-1)/2)
        self.conv = nn.Conv2d(in_channels_sum,out_channels,kernel_size=kernel_size,padding=padding)
        self.prelu = nn.PReLU(out_channels)
    
    def forward(self, x_fine_grained, x_upsampled):
        X = torch.cat((x_fine_grained, x_upsampled), dim=1)
        return self.prelu(self.conv(X))


class ResidualConvolution(nn.Module):
    def __init__(self, num_layers, num_channels,kernel_size=5):
        super(ResidualConvolution, self).__init__()
        padding = int((kernel_size-1)/2)

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.PReLU(num_channels))
        self.net = nn.Sequential(
            *layers
        )

        self.final_prelu = nn.PReLU(num_channels)


    def forward(self, x):
        conv_out = self.net(x)
        return self.final_prelu(conv_out + x)

class StackedConvolution(nn.Module):
    def __init__(self, num_layers, num_channels,kernel_size=5):
        super(StackedConvolution, self).__init__()
        padding = int((kernel_size-1)/2)

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.PReLU(num_channels))
        self.net = nn.Sequential(
            *layers
        )



    def forward(self, x):
        conv_out = self.net(x)
        return conv_out

class UpConvolution(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpConvolution, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
        self.p_relu = nn.PReLU(out_channels)

    def forward(self,x):
        return self.p_relu(self.up_conv(x))


class DownConvolution(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownConvolution, self).__init__()

        self.down_conv = nn.Conv2d(in_channels,out_channels,kernel_size=2,stride=2)
        self.p_relu = nn.PReLU(out_channels)

    def forward(self,x):
        return self.p_relu(self.down_conv(x))