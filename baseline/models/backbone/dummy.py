'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import torch.nn as nn

from baseline.models.registry import BACKBONE

@BACKBONE.register_module
class Dummy(nn.Module):
    def __init__(self,
                cfg=None):
        super(Dummy, self).__init__()
        self.cfg=cfg

    def forward(self, x):
        return x
