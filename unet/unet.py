import torch
import torch.nn as nn

from unet.unet_parts import DoubleConv, DownSample, UpSample

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dc_1 = DownSample(in_channels, 64)
        self.dc_2 = DownSample(64, 128)
        self.dc_3 = DownSample(128, 256)
        self.dc_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.uc_1 = UpSample(1024, 512)
        self.uc_2 = UpSample(512, 256)
        self.uc_3 = UpSample(256, 128)
        self.uc_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        d1, p1 = self.dc_1(x)
        d2, p2 = self.dc_2(p1)
        d3, p3 = self.dc_3(p2)
        d4, p4 = self.dc_4(p3)

        bn = self.bottle_neck(p4)

        u1 = self.uc_1(bn, d4)
        u2 = self.uc_2(u1, d3)
        u3 = self.uc_3(u2, d2)
        u4 = self.uc_4(u3, d1)

        return self.out(u4)