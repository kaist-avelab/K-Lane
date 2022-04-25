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
    head_type='row',
    loss_type='row_ce'
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
    type='VitSegNet', # GFC-T
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
    is_with_shared_mlp=False,
)

heads = dict(
    type='RowSharNotReducRef',
    dim_feat=8, # input feat channels
    row_size=144,
    dim_shared=512,
    lambda_cls=1.,
    thr_ext = 0.3,
    off_grid = 2,
    dim_token = 1024,
    tr_depth = 1,
    tr_heads = 16,
    tr_dim_head = 64,
    tr_mlp_dim = 2048,
    tr_dropout = 0.,
    tr_emb_dropout = 0.,
    is_reuse_same_network = False,
)

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
  lr = 0.0001,
)

epochs = 35
batch_size = 4
total_iter = (2904 // batch_size) * epochs
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = total_iter
)

eval_ep = 1
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
        mode_item='pc',
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        mode_item='pc',
    )
)
workers=12
