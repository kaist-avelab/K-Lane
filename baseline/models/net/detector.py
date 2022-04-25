import torch.nn as nn
import torch

from baseline.models.registry import NET
from ..registry import build_pcencoder, build_backbone, build_heads

@NET.register_module
class Detector(nn.Module):
    def __init__(self,
                head_type='seg',
                loss_type='row_ce',
                cfg=None):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(144, 144, 3, 3)
        self.pcencoder = build_pcencoder(cfg)
        self.backbone = build_backbone(cfg)
        self.heads = build_heads(cfg)
        self.head_type = head_type
        self.loss_type = loss_type

    def forward(self, batch, is_get_features=False):
        output = {}
        fea = self.pcencoder(batch)
        
        if is_get_features:
            fea, list_features = self.backbone(fea, True)
            output.update({'features': list_features})
        else:
            fea = self.backbone(fea)

        if self.training:
            out = self.heads(fea)
            output.update(self.heads.loss(out, batch, self.loss_type))
        else:
            out = self.heads(fea)
            if self.head_type == 'seg':
                output.update({'conf': out[:,7,:,:], 'cls': out[:,:7,:,:]})
                output.update({
                    'lane_maps': self.heads.get_lane_map_numpy_with_label(
                                        output, batch, is_img=self.cfg.view)})
            elif self.head_type == 'row':
                output.update(self.heads.get_conf_and_cls_dict(out, batch))
                output.update({
                    'lane_maps': self.heads.get_lane_map_numpy_with_label(
                                        output, batch, is_img=self.cfg.view)})

        return output
