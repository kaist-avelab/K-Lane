import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from baseline.models.registry import PCENCODER

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
}

# TODO is this acceptable registration?
@PCENCODER.register_module
class DenseProjector(nn.Module):
    def __init__(self,
                 densenet='densenet121',
                 pretrained=False,
                 replace_stride_with_dilation=[False, True, False],
                 out_conv=True,
                 in_channels=[64, 128, 256, -1],  # DenseNet121 block outputs
                 cfg=None):
        super(DenseProjector, self).__init__()
        self.cfg = cfg
        self.densenet = DenseNetWrapper(
            densenet=densenet,
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation,
            out_conv=out_conv,
            in_channels=in_channels,
            cfg=cfg
        )
        # if replace_stride_with_dilation[1]:
        #     self.model.transition2.pool = nn.Identity()
        # if replace_stride_with_dilation[2]:
        #     self.model.transition3.pool = nn.Identity()

    def forward(self, sample):
        proj = sample['proj']
        return self.densenet(proj)
    

class DenseNetWrapper(nn.Module):

    def __init__(self,
                 densenet='densenet121',
                 pretrained=False,
                 replace_stride_with_dilation=[False, True, False],
                 out_conv=False,
                 in_channels=[64, 128, 256, 1024],
                 cfg=None):
        super(DenseNetWrapper, self).__init__()
        from torchvision.models import densenet121

        # Load base DenseNet model
        if densenet == 'densenet121':
            self.model = densenet121(pretrained=pretrained).features
        else:
            raise NotImplementedError(f"DenseNet variant {densenet} not supported yet.")

        self.cfg = cfg
        self.in_channels = in_channels

        if out_conv:
            last_channel = next((c for c in reversed(in_channels) if c > 0), 1024)
            self.out = nn.Conv2d(last_channel, cfg.featuremap_out_channel, kernel_size=1, bias=False)
        else:
            self.out = None

        # optional?
        if replace_stride_with_dilation[1]:
            self.model.transition2.pool = nn.Identity()
        if replace_stride_with_dilation[2]:
            self.model.transition3.pool = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        if self.out:
            x = self.out(x)

        # print("[DEBUG] Final output shape:", x.shape)
        return x