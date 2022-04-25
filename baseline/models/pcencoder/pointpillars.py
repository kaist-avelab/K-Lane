import torch
import torch.nn as nn
import numpy as np

from baseline.models.registry import PCENCODER

@PCENCODER.register_module
class PointPillars(nn.Module):
    
    def __init__(self,
                max_points_per_pillar,
                num_features,
                num_channels,
                Xn, Yn,
                cfg=None):
        super(PointPillars, self).__init__()
        self.cfg=cfg
        
        self.pillar_encoder = nn.Sequential(
            nn.Conv2d(num_features, num_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.MaxPool2d((1, max_points_per_pillar))
        )
        self.Xn = Xn
        self.Yn = Yn
        self.num_channels = num_channels

    def forward(self, sample):
        pillars = sample['pillars']
        pillar_indices = sample['pillar_indices']
        
        pillars = pillars.permute(0, 3, 1, 2) # batch, features (channels), max pillars, max points
        
        out = self.pillar_encoder(pillars)
        
        # Send to CPU for scattering with numpy
        pillar_indices = pillar_indices.cpu()
        out = out.cpu()

        # Scattering
        batch_canvas = []
        batch_size = len(pillar_indices)
        for i in range(batch_size):
            canvas = torch.zeros((self.num_channels, self.Xn*self.Yn, 1))
            pillar_indices[i,:,2] = np.clip(pillar_indices[i,:,2], 0, 143)
            indices = (pillar_indices[i, :, 1] * self.Xn + pillar_indices[i,:,2])
            canvas[:, indices.long()] = out[i]
            batch_canvas.append(canvas)
        
        out = torch.stack(batch_canvas, 0)

        pillar_indices = pillar_indices.cuda()
        out = out.cuda()
        out = out.view(batch_size, self.num_channels, self.Xn, self.Yn)

        return out
