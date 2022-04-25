'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules import padding

from baseline.models.registry import BACKBONE

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        return out

class ResidualBlockCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, r = 0.5):
        super(ResidualBlockCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shared_mlp_cbam_1 = nn.Linear(out_channels, int(out_channels*r))
        self.shared_mlp_cbam_2 = nn.Linear(int(out_channels*r), out_channels)
        self.conv_cbam = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding = 3)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x)
        Fc_avg = x.mean(dim = -1).mean(dim = -1)
        Fc_max = x.max(dim = -1)[0].max(dim = -1)[0]
        Fc = torch.sigmoid(self.shared_mlp_cbam_2(torch.relu(self.shared_mlp_cbam_1(Fc_avg))) + 
                self.shared_mlp_cbam_2(torch.relu(self.shared_mlp_cbam_1(Fc_max))))
        
        Fc = Fc.unsqueeze(-1).unsqueeze(-1).repeat((1,1,H,W))
        Fc = torch.mul(x, Fc)

        Fs_avg = Fc.mean(dim = 1, keepdim=True)
        Fs_max = Fc.max(dim = 1, keepdim = True)[0]
        Fs = torch.sigmoid(self.conv_cbam(torch.cat((Fs_avg, Fs_max), dim = 1)))
        Fs = Fs.repeat((1, C, 1, 1))

        Fs = torch.mul(Fc, Fs)

        return (x + Fs)

@BACKBONE.register_module
class ResnetFPN(nn.Module):
    def __init__(self,
                num_channels,
                res_block=ResidualBlock,
                cfg=None):
        super(ResnetFPN, self).__init__()
        self.cfg = cfg
        
        # Block 1
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block1 = nn.Sequential(*block)

        # Block 2
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block2 = nn.Sequential(*block)

        # Block 3
        block = []
        block.append(nn.Conv2d(num_channels, 2*num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(2*num_channels))

        block.append(res_block(2*num_channels, 2*num_channels))
        block.append(res_block(2*num_channels, 2*num_channels))
        self.block3 = nn.Sequential(*block)

        # Block 4
        block = []
        block.append(nn.Conv2d(2*num_channels, 4*num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(4*num_channels))

        block.append(res_block(4*num_channels, 4*num_channels))
        block.append(res_block(4*num_channels, 4*num_channels))
        self.block4 = nn.Sequential(*block)

        # Block 5
        block = []
        block.append(nn.Conv2d(4*num_channels, 4*num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(4*num_channels))

        block.append(res_block(4*num_channels, 4*num_channels))
        block.append(res_block(4*num_channels, 4*num_channels))
        self.block5 = nn.Sequential(*block)

        # FPN
        self.up1 = nn.ConvTranspose2d(num_channels, 2*num_channels, kernel_size = 3, stride = 1, padding = 1)
        self.up2 = nn.ConvTranspose2d(num_channels, 2*num_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = (1,1))
        self.up3 = nn.ConvTranspose2d(2*num_channels, 2*num_channels, kernel_size = 3, stride = 4, padding = 1, output_padding = (3,3))
        self.up4 = nn.ConvTranspose2d(4*num_channels, 4*num_channels, kernel_size = 5, stride = 8, padding = 1, output_padding = (5,5))
        self.up5 = nn.ConvTranspose2d(4*num_channels, 4*num_channels, kernel_size = 9, stride = 16, padding = 1, output_padding = (9,9))

    def forward(self, x):
        ### Backbone ###
        x = self.block1(x)
        up_1 = self.up1(x)

        x = self.block2(x)
        up_2 = self.up2(x)

        x = self.block3(x)
        up_3 = self.up3(x)

        x = self.block4(x)
        up_4 = self.up4(x)

        x = self.block5(x)
        up_5 = self.up5(x)

        ### Neck ### 
        out = torch.cat((up_1, up_2, up_3, up_4, up_5),1)

        return out

@BACKBONE.register_module
class ResnetFPN2(nn.Module):
    def __init__(self,
                num_channels,
                res_block=ResidualBlock,
                cfg=None):
        super(ResnetFPN2, self).__init__()
        self.cfg = cfg
        
        # Block 1
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block1 = nn.Sequential(*block)

        # Block 2
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block2 = nn.Sequential(*block)

        # FPN
        self.up1 = nn.ConvTranspose2d(num_channels, 2*num_channels, kernel_size = 3, stride = 1, padding = 1)
        self.up2 = nn.ConvTranspose2d(num_channels, 2*num_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = (1,1))

    def forward(self, x):
        ### Backbone ###
        x = self.block1(x)
        up_1 = self.up1(x)

        x = self.block2(x)
        up_2 = self.up2(x)

        ### Neck ### 
        out = torch.cat((up_1, up_2),1)

        return out

@BACKBONE.register_module
class ResnetFPN3(nn.Module):
    def __init__(self,
                num_channels,
                res_block=ResidualBlock,
                cfg=None):
        super(ResnetFPN3, self).__init__()
        self.cfg = cfg
        
        # Block 1
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block1 = nn.Sequential(*block)

        # Block 2
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block2 = nn.Sequential(*block)

        # Block 3
        block = []
        block.append(nn.Conv2d(num_channels, 2*num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(2*num_channels))

        block.append(res_block(2*num_channels, 2*num_channels))
        block.append(res_block(2*num_channels, 2*num_channels))
        self.block3 = nn.Sequential(*block)


        # FPN
        self.up1 = nn.ConvTranspose2d(num_channels, 2*num_channels, kernel_size = 3, stride = 1, padding = 1)
        self.up2 = nn.ConvTranspose2d(num_channels, 2*num_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = (1,1))
        self.up3 = nn.ConvTranspose2d(2*num_channels, 2*num_channels, kernel_size = 3, stride = 4, padding = 1, output_padding = (3,3))

    def forward(self, x):
        ### Backbone ###
        x = self.block1(x)
        up_1 = self.up1(x)

        x = self.block2(x)
        up_2 = self.up2(x)

        x = self.block3(x)
        up_3 = self.up3(x)


        ### Neck ### 
        out = torch.cat((up_1, up_2, up_3),1)

        return out

@BACKBONE.register_module
class ResnetFPN4(nn.Module):
    def __init__(self,
                num_channels,
                res_block=ResidualBlock,
                cfg=None):
        super(ResnetFPN4, self).__init__()
        self.cfg = cfg
        
        # Block 1
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block1 = nn.Sequential(*block)

        # Block 2
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block2 = nn.Sequential(*block)

        # Block 3
        block = []
        block.append(nn.Conv2d(num_channels, 2*num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(2*num_channels))

        block.append(res_block(2*num_channels, 2*num_channels))
        block.append(res_block(2*num_channels, 2*num_channels))
        self.block3 = nn.Sequential(*block)

        # Block 4
        block = []
        block.append(nn.Conv2d(2*num_channels, 4*num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(4*num_channels))

        block.append(res_block(4*num_channels, 4*num_channels))
        block.append(res_block(4*num_channels, 4*num_channels))
        self.block4 = nn.Sequential(*block)


        # FPN
        self.up1 = nn.ConvTranspose2d(num_channels, 2*num_channels, kernel_size = 3, stride = 1, padding = 1)
        self.up2 = nn.ConvTranspose2d(num_channels, 2*num_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = (1,1))
        self.up3 = nn.ConvTranspose2d(2*num_channels, 2*num_channels, kernel_size = 3, stride = 4, padding = 1, output_padding = (3,3))
        self.up4 = nn.ConvTranspose2d(4*num_channels, 4*num_channels, kernel_size = 5, stride = 8, padding = 1, output_padding = (5,5))

    def forward(self, x):
        ### Backbone ###
        x = self.block1(x)
        up_1 = self.up1(x)

        x = self.block2(x)
        up_2 = self.up2(x)

        x = self.block3(x)
        up_3 = self.up3(x)

        x = self.block4(x)
        up_4 = self.up4(x)


        ### Neck ### 
        out = torch.cat((up_1, up_2, up_3, up_4),1)

        return out

@BACKBONE.register_module
class ResnetFPN3_CBAM(nn.Module):
    def __init__(self,
                num_channels,
                res_block=ResidualBlockCBAM,
                cfg=None):
        super(ResnetFPN3_CBAM, self).__init__()
        self.cfg = cfg
        
        # Block 1
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block1 = nn.Sequential(*block)

        # Block 2
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block2 = nn.Sequential(*block)

        # Block 3
        block = []
        block.append(nn.Conv2d(num_channels, 2*num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(2*num_channels))

        block.append(res_block(2*num_channels, 2*num_channels))
        block.append(res_block(2*num_channels, 2*num_channels))
        self.block3 = nn.Sequential(*block)


        # FPN
        self.up1 = nn.ConvTranspose2d(num_channels, 2*num_channels, kernel_size = 3, stride = 1, padding = 1)
        self.up2 = nn.ConvTranspose2d(num_channels, 2*num_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = (1,1))
        self.up3 = nn.ConvTranspose2d(2*num_channels, 2*num_channels, kernel_size = 3, stride = 4, padding = 1, output_padding = (3,3))

    def forward(self, x):
        ### Backbone ###
        x = self.block1(x)
        up_1 = self.up1(x)

        x = self.block2(x)
        up_2 = self.up2(x)

        x = self.block3(x)
        up_3 = self.up3(x)


        ### Neck ### 
        out = torch.cat((up_1, up_2, up_3),1)

        return out

@BACKBONE.register_module
class ResnetFPN4_CBAM(nn.Module):
    def __init__(self,
                num_channels,
                res_block=ResidualBlockCBAM,
                cfg=None):
        super(ResnetFPN4_CBAM, self).__init__()
        self.cfg = cfg
        
        # Block 1
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block1 = nn.Sequential(*block)

        # Block 2
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block2 = nn.Sequential(*block)

        # Block 3
        block = []
        block.append(nn.Conv2d(num_channels, 2*num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(2*num_channels))

        block.append(res_block(2*num_channels, 2*num_channels))
        block.append(res_block(2*num_channels, 2*num_channels))
        self.block3 = nn.Sequential(*block)

        # Block 4
        block = []
        block.append(nn.Conv2d(2*num_channels, 4*num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(4*num_channels))

        block.append(res_block(4*num_channels, 4*num_channels))
        block.append(res_block(4*num_channels, 4*num_channels))
        self.block4 = nn.Sequential(*block)


        # FPN
        self.up1 = nn.ConvTranspose2d(num_channels, 2*num_channels, kernel_size = 3, stride = 1, padding = 1)
        self.up2 = nn.ConvTranspose2d(num_channels, 2*num_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = (1,1))
        self.up3 = nn.ConvTranspose2d(2*num_channels, 2*num_channels, kernel_size = 3, stride = 4, padding = 1, output_padding = (3,3))
        self.up4 = nn.ConvTranspose2d(4*num_channels, 4*num_channels, kernel_size = 5, stride = 8, padding = 1, output_padding = (5,5))

    def forward(self, x):
        ### Backbone ###
        x = self.block1(x)
        up_1 = self.up1(x)

        x = self.block2(x)
        up_2 = self.up2(x)

        x = self.block3(x)
        up_3 = self.up3(x)

        x = self.block4(x)
        up_4 = self.up4(x)


        ### Neck ### 
        out = torch.cat((up_1, up_2, up_3, up_4),1)

        return out


@BACKBONE.register_module
class ResnetFPN2_Dilated(nn.Module):
    def __init__(self,
                num_channels,
                res_block=ResidualBlock,
                cfg=None):
        super(ResnetFPN2_Dilated, self).__init__()
        self.cfg = cfg
        
        # Block 1
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block1 = nn.Sequential(*block)

        # Block 2
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 2, dilation=2))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block2 = nn.Sequential(*block)

    def forward(self, x):
        ### Backbone ###
        x1 = self.block1(x)
        x2 = self.block2(x1)

        ### Neck ### 
        out = torch.cat((x1, x2),1)

        return out

@BACKBONE.register_module
class ResnetFPN3_Dilated(nn.Module):
    def __init__(self,
                num_channels,
                res_block=ResidualBlock,
                cfg=None):
        super(ResnetFPN3, self).__init__()
        self.cfg = cfg
        
        # Block 1
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block1 = nn.Sequential(*block)

        # Block 2
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 2, dilation=2))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block2 = nn.Sequential(*block)

        # Block 3
        block = []
        block.append(nn.Conv2d(num_channels, 2*num_channels, kernel_size = 3, stride = 1, padding = 2, dilation = 2))
        block.append(nn.BatchNorm2d(2*num_channels))

        block.append(res_block(2*num_channels, 2*num_channels))
        block.append(res_block(2*num_channels, 2*num_channels))
        self.block3 = nn.Sequential(*block)

    def forward(self, x):
        ### Backbone ###
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)


        ### Neck ### 
        out = torch.cat((x1, x2, x3),1)

        return out

@BACKBONE.register_module
class ResnetFPN4_Dilated(nn.Module):
    def __init__(self,
                num_channels,
                res_block=ResidualBlock,
                cfg=None):
        super(ResnetFPN4_Dilated, self).__init__()
        self.cfg = cfg
        
        # Block 1
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block1 = nn.Sequential(*block)

        # Block 2
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 2, dilation = 2))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(res_block(num_channels, num_channels))
        block.append(res_block(num_channels, num_channels))
        self.block2 = nn.Sequential(*block)

        # Block 3
        block = []
        block.append(nn.Conv2d(num_channels, 2*num_channels, kernel_size = 3, stride = 1, padding = 2, dilation = 2))
        block.append(nn.BatchNorm2d(2*num_channels))

        block.append(res_block(2*num_channels, 2*num_channels))
        block.append(res_block(2*num_channels, 2*num_channels))
        self.block3 = nn.Sequential(*block)

        # Block 4
        block = []
        block.append(nn.Conv2d(2*num_channels, 4*num_channels, kernel_size = 3, stride = 1, padding = 2, dilation = 2))
        block.append(nn.BatchNorm2d(4*num_channels))

        block.append(res_block(4*num_channels, 4*num_channels))
        block.append(res_block(4*num_channels, 4*num_channels))
        self.block4 = nn.Sequential(*block)

    def forward(self, x):
        ### Backbone ###
        x1 = self.block1(x)

        x2 = self.block2(x1)

        x3 = self.block3(x2)

        x4 = self.block4(x3)


        ### Neck ### 
        out = torch.cat((x1, x2, x3, x4),1)

        return out
