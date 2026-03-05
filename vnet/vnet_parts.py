import torch
import torch.nn as nn


class InputConvolution(nn.Module):
    def __init__(self, out_channels=16):
        super(InputConvolution, self).__init__()
        self.conv = nn.Conv2d(1, out_channels, kernel_size=5, padding=2)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        out = self.conv(x) 
        x_repeat = x.expand(-1, out.shape[1], -1, -1)

        return self.prelu(out + x_repeat)

class ResidualConvolution(nn.Module):
    def __init__(self, num_layers, num_channels):
        super(ResidualConvolution, self).__init__()


        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=5, padding=2))
            layers.append(nn.PReLU(num_channels))
        self.net = nn.Sequential(
            *layers
        )

        self.final_prelu = nn.PReLU(num_channels)


    def forward(self, x):
        conv_out = self.net(x)
        return self.final_prelu(conv_out + x)

class StackedConvolution(nn.Module):
    def __init__(self, num_layers, num_channels):
        super(StackedConvolution, self).__init__()


        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=5, padding=2))
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
