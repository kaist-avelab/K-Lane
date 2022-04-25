'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import torch
import torch.nn as nn
import numpy as np

from baseline.models.registry import BACKBONE

class VggBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VggBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        return out

@BACKBONE.register_module
class VggFPN(nn.Module):
    def __init__(self,
                num_channels,
                vgg_block=VggBlock,
                cfg=None):
        super(VggFPN, self).__init__()
        self.cfg = cfg
        
        # Block 1
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(vgg_block(num_channels, num_channels))
        self.block1 = nn.Sequential(*block)

        # Block 2
        block = []
        block.append(nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(num_channels))
        block.append(vgg_block(num_channels, num_channels))
        block.append(vgg_block(num_channels, num_channels))
        self.block2 = nn.Sequential(*block)

        # Block 3
        block = []
        block.append(nn.Conv2d(num_channels, 2*num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(2*num_channels))

        block.append(vgg_block(2*num_channels, 2*num_channels))
        block.append(vgg_block(2*num_channels, 2*num_channels))
        self.block3 = nn.Sequential(*block)

        # Block 4
        block = []
        block.append(nn.Conv2d(2*num_channels, 4*num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(4*num_channels))

        block.append(vgg_block(4*num_channels, 4*num_channels))
        block.append(vgg_block(4*num_channels, 4*num_channels))
        self.block4 = nn.Sequential(*block)

        # Block 5
        block = []
        block.append(nn.Conv2d(4*num_channels, 4*num_channels, kernel_size = 3, stride = 2, padding = 1))
        block.append(nn.BatchNorm2d(4*num_channels))

        block.append(vgg_block(4*num_channels, 4*num_channels))
        block.append(vgg_block(4*num_channels, 4*num_channels))
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
