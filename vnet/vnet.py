import torch
import torch.nn as nn
from vnet_parts import *

class VNet2D(nn.Module):
    
    def __init__(self):
        super(VNet2D,self).__init__()

        self.input_conv = InputConvolution()
        
        self.d_rs_1 = ResidualConvolution(1,16)
        self.down_1 = DownConvolution(16, 32)
        self.d_rs_2 = ResidualConvolution(2, 32)
        self.down_2 = DownConvolution(32, 64)
        self.d_rs_3 = ResidualConvolution(3, 64)
        self.down_3 = DownConvolution(64, 128)
        self.d_rs_4 = ResidualConvolution(3, 128)
        self.down_4 = DownConvolution(128, 256)

        self.bottleneck = ResidualConvolution(3, 256)

        self.up_1 = UpConvolution(256, 256)
        self.u_rs_1 = StackedConvolution(3, 256)
        self.p_relu_1 = nn.PReLU(256)

        self.up_2 = UpConvolution(256, 128)
        self.u_rs_2 = StackedConvolution(3, 128)
        self.p_relu_2 = nn.PReLU(128)

        self.up_3 = UpConvolution(128, 64)
        self.u_rs_3 = StackedConvolution(2, 64)
        self.p_relu_3 = nn.PReLU(64)

        self.up_4 = UpConvolution(64, 32)
        self.u_rs_4 = StackedConvolution(1, 32)
        self.p_relu_4 = nn.PReLU(32)


        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)


    def forward(self,x):

        
        inp_out = self.input_conv(x)

        out_16 = self.d_rs_1(inp_out)
        
        out_32 = self.down_1(out_16)
        out_32 = self.d_rs_2(out_32)
        
        out_64 = self.down_2(out_32)
        out_64 = self.d_rs_3(out_64)
        
        out_128 = self.down_3(out_64)
        out_128 = self.d_rs_4(out_128)
        
        out_256 = self.down_4(out_128)
        

        out_b = self.bottleneck(out_256)
        

        out_u1 = self.up_1(out_b)
        out_256_u = self.u_rs_1(out_u1 + out_256)
        out_256_prelu = self.p_relu_1(out_256_u + out_u1)
        
        out_u2 = self.up_2(out_256_prelu)
        out_128_u = self.u_rs_2(out_u2 + out_128)
        out_128_prelu = self.p_relu_2(out_128_u + out_u2)

        
        out_u3 = self.up_3(out_128_prelu)
        out_64_u = self.u_rs_3(out_u3 + out_64)
        out_64_prelu = self.p_relu_3(out_64_u + out_u3)
        
        out_u4 = self.up_4(out_64_prelu)
        out_32_u = self.u_rs_4(out_u4 + out_32)
        out_32_prelu = self.p_relu_4(out_32_u + out_u4)

        
        return self.out_conv(out_32_prelu)