# Thanks to TuZheng (LaneDet) https://github.com/Turoad/lanedet
from baseline.utils import Registry, build_from_cfg
import torch.nn as nn

PCENCODER = Registry('pcencoder')
BACKBONE = Registry('backbone')
HEADS = Registry('heads')
NET = Registry('net')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_pcencoder(cfg):
    return build(cfg.pcencoder, PCENCODER, default_args=dict(cfg=cfg))

def build_backbone(cfg):
    return build(cfg.backbone, BACKBONE, default_args=dict(cfg=cfg))

def build_heads(cfg):
    return build(cfg.heads, HEADS, default_args=dict(cfg=cfg))

def build_head(split_cfg, cfg):
    return build(split_cfg, HEADS, default_args=dict(cfg=cfg))

def build_net(cfg):
    return build(cfg.net, NET, default_args=dict(cfg=cfg))
