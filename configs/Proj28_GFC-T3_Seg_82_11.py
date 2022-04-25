'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
seed = 2021
load_from = None
finetune_from = None
log_dir = './logs'
view = False

net = dict(
    type='Detector',
)

pcencoder = dict(
    type='Projector',
    resnet='resnet34',
    pretrained=False,
    replace_stride_with_dilation=[False, True, False],
    out_conv=True,
    in_channels=[64, 128, 256, -1]
)
featuremap_out_channel = 64

filter_mode = 'xyz'
list_filter_roi = [0.02, 46.08, -11.52, 11.52, -2.0, 1.5]  # get rid of 0, 0 points
list_roi_xy = [0.0, 46.08, -11.52, 11.52]
list_grid_xy = [0.04, 0.02]
list_img_size_xy = [1152, 1152]

backbone = dict(
    type='VitSegNet',
    image_size=144,
    patch_size=8,
    channels=64,
    dim=512,
    depth=3,
    heads=16,
    output_channels=1024,
    expansion_factor=4,
    dim_head=64,
    dropout=0.,
    emb_dropout=0., # TBD
)

heads = dict(
    type='GridSeg',
    num_1=1024,
    num_2=2048,
    num_classes=7,
)
conf_thr = 0.5
view = True
conf_thr = 0.5
view = True

# BGR Format to OpenCV
cls_lane_color = [
    (0, 0, 255),
    (0, 50, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 100)
]

optimizer = dict(
  type = 'Adam', #'AdamW',
  lr = 0.0002,
)

epochs = 60
batch_size = 4
total_iter = (7687 // batch_size) * epochs
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = total_iter
)

eval_ep = 2
save_ep = 1

### Setting Here ###
dataset_path = './data/KLane' # '/media/donghee/HDD_0/KLane'
### Setting Here ###
dataset_type = 'KLane'
dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='train',
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
    )
)
workers=12
