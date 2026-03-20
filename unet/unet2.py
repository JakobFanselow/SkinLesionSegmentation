import torch
import torch.nn as nn

from unet.unet_parts import DoubleConv, DownSample, UpSample, AttentionGate, ResidualBlock, RecConvBlock, R2ConvBlock

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_filters=64, depth=4, activation='relu', bottle_neck=True, kernel_size=3):
        super().__init__()
        
        self.init_filters = init_filters
        self.depth = depth
        self.activation = activation
        self.use_bottleneck = bottle_neck
        self.kernel_size = kernel_size
        
        fn_args = {
            "kernel_size": self.kernel_size, 
            "activation": self.activation
        }
        
        self.encoder = nn.ModuleList()
        in_ch = in_channels
        out_ch = init_filters
        for _ in range(depth):
            self.encoder.append(DownSample(in_ch, out_ch, **fn_args))
            in_ch = out_ch
            out_ch *= 2
        if self.use_bottleneck:
            self.bottleneck = DoubleConv(in_ch, in_ch * 2, **fn_args)
            decode_ch = in_ch * 2
        else:
            self.bottleneck = nn.Identity()
            decode_ch = in_ch
        
        self.decoder = nn.ModuleList()
        in_ch = decode_ch
        for level in reversed(range(depth)):
            skip_ch = init_filters * (2 ** level)
            self.decoder.append(UpSample(in_ch, skip_ch, **fn_args))
            in_ch = skip_ch
        
        self.out = nn.Conv2d(in_channels=init_filters, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        p = x
        for down in self.encoder:
            d, p = down(p)
            skips.append(d)
        
        x = self.bottleneck(p)
        
        for up, skip in zip(self.decoder, reversed(skips)):
            x = up(x, skip)
        
        return self.out(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dc_1 = DownSample(in_channels, 64)
        self.dc_2 = DownSample(64, 128)
        self.dc_3 = DownSample(128, 256)
        self.dc_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.ag1 = AttentionGate(F_g=1024, F_l=512, F_int=256)
        self.ag2 = AttentionGate(F_g=512,  F_l=256, F_int=128)
        self.ag3 = AttentionGate(F_g=256,  F_l=128, F_int=64)
        self.ag4 = AttentionGate(F_g=128,  F_l=64,  F_int=32)

        self.uc_1 = UpSample(1024, 512)
        self.uc_2 = UpSample(512, 256)
        self.uc_3 = UpSample(256, 128)
        self.uc_4 = UpSample(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1, p1 = self.dc_1(x)
        d2, p2 = self.dc_2(p1)
        d3, p3 = self.dc_3(p2)
        d4, p4 = self.dc_4(p3)

        bn = self.bottle_neck(p4)

        s4 = self.ag1(g=bn, x=d4)
        u1 = self.uc_1(bn, s4)

        s3 = self.ag2(g=u1, x=d3)
        u2 = self.uc_2(u1, s3)

        s2 = self.ag3(g=u2, x=d2)
        u3 = self.uc_3(u2, s2)

        s1 = self.ag4(g=u3, x=d1)
        u4 = self.uc_4(u3, s1)

        return self.out(u4)

class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ResidualBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ResidualBlock(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        bn = self.bottleneck(p4)

        u1 = self.up1(bn)
        u1 = torch.cat([u1, e4], dim=1)
        d1 = self.dec1(u1)
        
        u2 = self.up2(d1)
        u2 = torch.cat([u2, e3], dim=1)
        d2 = self.dec2(u2)
        
        u3 = self.up3(d2)
        u3 = torch.cat([u3, e2], dim=1)
        d3 = self.dec3(u3)
        
        u4 = self.up4(d3)
        u4 = torch.cat([u4, e1], dim=1)
        d4 = self.dec4(u4)

        return self.out(d4)

class RecUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, t=2):
        super().__init__()
        
        self.enc1 = RecConvBlock(in_channels, 64, t=t)
        self.enc2 = RecConvBlock(64, 128, t=t)
        self.enc3 = RecConvBlock(128, 256, t=t)
        self.enc4 = RecConvBlock(256, 512, t=t)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = RecConvBlock(512, 1024, t=t)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = RecConvBlock(1024, 512, t=t)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = RecConvBlock(512, 256, t=t)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = RecConvBlock(256, 128, t=t)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = RecConvBlock(128, 64, t=t)
        
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        bn = self.bottleneck(self.pool(e4))
        
        d4 = self.up4(bn)
        d4 = self.dec4(torch.cat((e4, d4), dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat((e3, d3), dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat((e2, d2), dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat((e1, d1), dim=1))
        
        return self.out_conv(d1)
    
class R2UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, t=2):
        super().__init__()
        
        self.enc1 = R2ConvBlock(in_channels, 64, t=t)
        self.enc2 = R2ConvBlock(64, 128, t=t)
        self.enc3 = R2ConvBlock(128, 256, t=t)
        self.enc4 = R2ConvBlock(256, 512, t=t)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = R2ConvBlock(512, 1024, t=t)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = R2ConvBlock(1024, 512, t=t)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = R2ConvBlock(512, 256, t=t)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = R2ConvBlock(256, 128, t=t)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = R2ConvBlock(128, 64, t=t)
        
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        bn = self.bottleneck(self.pool(e4))
        
        d4 = self.up4(bn)
        d4 = self.dec4(torch.cat((e4, d4), dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat((e3, d3), dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat((e2, d2), dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat((e1, d1), dim=1))
        
        return self.out_conv(d1) 