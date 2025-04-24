'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import torch
import torch.nn as nn
import numpy as np
import cv2
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from baseline.models.registry import HEADS

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

"""
Applies layer normalization before a submodule.

dim: number of features in the last dimension of the input tensor
dn: some function or module that operates on a normalized input
"""
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

"""
Standard position-wise feed forward network. Applies two linear layers with
non-linearity in between. The "MLP" of a transformer block, applied after attention
to allow nonlinear transformation and feature mixing within each token embedding.

dim: dimensionaliy of the input and output
hidden_dim: intermediate dimensionality fr MLP expansion
"""
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), # project input to higher dimension
            nn.GELU(),                  # nonlinear activation
            nn.Linear(hidden_dim, dim), # project back to original dimension
        )

    def forward(self, x):
        return self.net(x)
    
"""
Defines the multi-head self attention mechanism. Applies scaled dot product with mutliple
heads, allowing model to capture relationships between tokens.

dim: dimension of input features
heads: number of attention heads
dim_head: dimensionality per head
"""
class Attention(nn.Module):
    def __init__(self, dim, heads=2, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
"""
Standard transformer encoder block implemented with prenorm architecture and 
residual connections.

dim: input and output dimensionality of the transformer
depth: number of repeated transformer layers
heads: number of attention heads
dim_head: dimensionality of each head
mlp_dim: hidden dimension of the FFN
"""
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim))
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
"""
Lightweight row-wise transformer.
"""    
@HEADS.register_module
class LightRowTransformer(nn.Module):
    def __init__(self, dim_feat=8, row_size=144, dim_token=256, num_cls=6, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.row_tensor_maker = rearrange
        self.num_cls = num_cls
        self.token_window = 5  # 2*off_grid+1
        self.off_grid = 2
        in_token_channel = dim_feat * row_size * self.token_window

        # Shared 1D heads
        self.shared_ext_head = nn.Sequential(
            nn.Conv1d(dim_feat * row_size, 256, 1),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 2, 1),
            rearrange('b c h -> b h c')
        )

        self.shared_cls_head = nn.Sequential(
            nn.Conv1d(dim_feat * row_size, 256, 1),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, row_size, 1),
            rearrange('b w h -> b h w')
        )

        self.to_token = nn.Sequential(
            rearrange('c h w -> (c h w)'),
            nn.Linear(in_token_channel, dim_token)
        )

        self.embeddings = nn.Parameter(torch.randn(num_cls, dim_token))

        self.transformer = nn.Sequential(
            Transformer(dim_token, depth=1, heads=2, dim_head=32, mlp_dim=512),
            nn.LayerNorm(dim_token),
            nn.Linear(dim_token, in_token_channel),
            rearrange('b n (c h w) -> b n c h w', c=dim_feat, h=row_size, w=self.token_window)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        row_tensor = rearrange(x, 'b c h w -> b (c w) h')

        ext = torch.softmax(self.shared_ext_head(row_tensor), dim=2)  # (B, H, 2)
        cls = torch.softmax(self.shared_cls_head(row_tensor), dim=2)  # (B, H, W)

        x_padded = torch.cat([
            torch.zeros((B, C, H, self.off_grid), device=x.device),
            x,
            torch.zeros((B, C, H, self.off_grid), device=x.device)
        ], dim=3)

        tokens = []
        coords_list = []
        for cls_idx in range(self.num_cls):
            if torch.mean(ext[:, :, 0]) > 0.3:  # fake selection threshold
                argmax_col = torch.argmax(cls, dim=2)  # (B, H)
                token_batch = []
                for b in range(B):
                    patches = torch.zeros((C, H, self.token_window), device=x.device)
                    for h in range(H):
                        center = argmax_col[b, h] + self.off_grid
                        patches[:, h, :] = x_padded[b, :, h, center - self.off_grid:center + self.off_grid + 1]
                    token = self.to_token(patches) + self.embeddings[cls_idx]
                    token_batch.append(token.unsqueeze(0))
                tokens.append(torch.cat(token_batch, dim=0).unsqueeze(1))
                coords_list.append(argmax_col)

        if tokens:
            tokens = torch.cat(tokens, dim=1)
            refined = self.transformer(tokens)
            for i, coords in enumerate(coords_list):
                for b in range(B):
                    for h in range(H):
                        center = coords[b, h] + self.off_grid
                        x_padded[b, :, h, center - self.off_grid:center + self.off_grid + 1] = refined[b, i, :, h, :]

        x = x_padded[:, :, :, self.off_grid:W + self.off_grid]
        row_tensor = rearrange(x, 'b c h w -> b (c w) h')

        ext2 = torch.softmax(self.shared_ext_head(row_tensor), dim=2)
        cls2 = torch.softmax(self.shared_cls_head(row_tensor), dim=2)

        return {
            'ext': ext,
            'cls': cls,
            'ext2': ext2,
            'cls2': cls2
        }
    

    ### --- From other --- ###
    
    def label_formatting(self, raw_label, is_get_label_as_tensor = False):
        # Output image: top-left of the image is farthest-left
        num_of_labels = len(raw_label)
        label_tensor = np.zeros((num_of_labels, 2, 144, 144), dtype = np.longlong)

        for k in range(num_of_labels):
            label_temp = np.zeros((144,144,2), dtype = np.longlong)
            label_data = raw_label[k]

            for i in range(144):
                for j in range(144):

                    y_idx = 144 - i - 1
                    x_idx = 144 - j - 1

                    line_num = int(label_data[i][j])
                    if line_num == 255:
                        label_temp[y_idx][x_idx][1] = 0
                        # classification
                        label_temp[y_idx][x_idx][0] = 6
                    else: # class
                        # confidence
                        label_temp[y_idx][x_idx][1] = 1
                        # classification
                        label_temp[y_idx][x_idx][0] = line_num

            label_tensor[k,:,:,:] = np.transpose(label_temp, (2, 0, 1))

        if is_get_label_as_tensor:
            return torch.tensor(label_tensor)
        else:
            return label_tensor

    def get_lane_exist_and_cls_wise_maps(self, label_tensor, is_one_hot=False, is_ret_list=True):
        b, _, img_h, img_w = label_tensor.shape # _, 2, 144, 144
        n_cls = self.num_cls

        ### Vis each batch ###
        # temp_conf_tensor = np.squeeze(label_tensor[0,1,:,:])
        # temp_cls_tensor = np.squeeze(label_tensor[0,0,:,:])

        # temp_conf_vis_img = np.zeros((144,144), dtype=np.uint8)
        # temp_conf_vis_img[np.where(temp_conf_tensor==1)] = 255

        # temp_cls_vis_img_2 = np.zeros((144,144), dtype=np.uint8)
        # temp_cls_vis_img_2[np.where(temp_cls_tensor==2)] = 255

        # temp_cls_vis_img_3 = np.zeros((144,144), dtype=np.uint8)
        # temp_cls_vis_img_3[np.where(temp_cls_tensor==3)] = 255

        # print(np.unique(temp_conf_tensor))
        # print(np.unique(temp_cls_tensor))

        # cv2.imshow('temp', temp_conf_vis_img)
        # cv2.imshow('temp_2', temp_cls_vis_img_2)
        # cv2.imshow('temp_3', temp_cls_vis_img_3)
        # cv2.waitKey(0)
        ### Vis each batch ###

        lb_cls = np.squeeze(label_tensor[:,0,:,:]) # 0~5: cls, 6: background
        
        ret_exist = np.zeros((b,n_cls,img_h)) # 2,6,144
        ret_maps = np.zeros((b,n_cls,img_h,img_w))
        for idx_b in range(b):
            ret_exist[idx_b,:,:], ret_maps[idx_b,:,:,:] = \
                self.get_line_existence_and_cls_wise_maps_per_batch(np.squeeze(lb_cls[idx_b,:,:]))
        
        if is_one_hot:
            ret_ext_oh = np.zeros((b,n_cls,img_h,2))
            ret_ext_oh[:,:,:,0][np.where(ret_exist==1.)] = 1.
            ret_ext_oh[:,:,:,1][np.where(ret_exist==0.)] = 1.
            ret_exist = ret_ext_oh

        if is_ret_list:
            list_ext = []
            list_cls = []
            
            for idx_cls in range(self.num_cls):
                list_ext.append(torch.tensor(np.squeeze(ret_exist[:,idx_cls,:,:])).cuda())
                list_cls.append(torch.tensor(np.squeeze(ret_maps[:,idx_cls,:,:])).cuda())
            
            return list_ext, list_cls
        else:
            return ret_exist, ret_maps

    def get_line_existence_and_cls_wise_maps_per_batch(self, lb_cls, n_cls=6, img_h=144, img_w=144):
        # print(lb_cls.shape)
        cls_maps = np.zeros((n_cls,img_h,img_w))
        line_ext = np.zeros((n_cls,img_h))
        for idx_cls in range(n_cls):
            cls_maps[idx_cls,:,:][np.where(lb_cls==idx_cls)] = 1.
            lb_conf = np.zeros_like(lb_cls)
            lb_conf[np.where(lb_cls==idx_cls)]=1.
            line_ext[idx_cls,:] = np.sum(lb_conf, axis=1)
            # print(line_ext[idx_cls,:])
            # print(line_ext[idx_cls,:].shape)

        # for i in range(6):
        #     cv2.imshow(f'hi_{i}', cls_maps[i,:,:])
        # cv2.waitKey(0)
        
        return line_ext, cls_maps

    def get_conf_and_cls_dict(self, out, is_get_1_stage_result=True):
        # b, 144, 144 conf
        # b, 7, 144, 144 cls
        b_size, img_h, img_w = out['cls_2'].shape
        num_cls = self.num_cls
        idx_bg = num_cls

        arr_conf = np.zeros((b_size, img_h, img_w))
        arr_cls = np.zeros((b_size, num_cls+1, img_h, img_w))

        for idx_b in range(b_size):
            for idx_c in range(num_cls):
                # exist: one-hot to 0 (not), 1 (line)
                temp_ext = torch.argmax(out[f'ext2_{idx_c}'][idx_b,:], dim=1).detach().cpu().numpy()
                # print(temp_ext)
                # print(temp_ext.shape)
                for idx_h in range(img_h):
                    if temp_ext[idx_h] == 0.:
                        # print(out[f'cls_{idx_c}'][idx_b,idx_h,:].shape)
                        corr_idx = torch.argmax(out[f'cls2_{idx_c}'][idx_b,idx_h,:]).detach().cpu().item()
                        # print(corr_idx)
                        arr_cls[idx_b, idx_c,  idx_h, corr_idx] = 1. # foreground
                        arr_cls[idx_b, idx_bg, idx_h, corr_idx] = 1. # background
                    else: # lane not exist
                        continue
        
        # print(arr_cls[:,idx_bg,:,:].shape)
        arr_conf[np.where(arr_cls[:,idx_bg,:,:]==1.)] = 1.
        dict_ret = {'conf': torch.tensor(arr_conf), 'cls': torch.tensor(arr_cls)}

        if is_get_1_stage_result:
            b_size, img_h, img_w = out['cls_2'].shape # ego_lane
            num_cls = self.num_cls
            idx_bg = num_cls

            arr_conf = np.zeros((b_size, img_h, img_w))
            arr_cls = np.zeros((b_size, num_cls+1, img_h, img_w))

            for idx_b in range(b_size):
                for idx_c in range(num_cls):
                    # exist: one-hot to 0 (not), 1 (line)
                    temp_ext = torch.argmax(out[f'ext_{idx_c}'][idx_b,:], dim=1).detach().cpu().numpy()
                    # print(temp_ext)
                    # print(temp_ext.shape)
                    for idx_h in range(img_h):
                        if temp_ext[idx_h] == 0.:
                            # print(out[f'cls_{idx_c}'][idx_b,idx_h,:].shape)
                            corr_idx = torch.argmax(out[f'cls_{idx_c}'][idx_b,idx_h,:]).detach().cpu().item()
                            # print(corr_idx)
                            arr_cls[idx_b, idx_c,  idx_h, corr_idx] = 1. # foreground
                            arr_cls[idx_b, idx_bg, idx_h, corr_idx] = 1. # background
                        else: # lane not exist
                            continue
            
            # print(arr_cls[:,idx_bg,:,:].shape)
            arr_conf[np.where(arr_cls[:,idx_bg,:,:]==1.)] = 1.

            dict_ret.update({'conf_1': torch.tensor(arr_conf), 'cls_1': torch.tensor(arr_cls)})

        return dict_ret

    def loss(self, out, batch, loss_type=None):
        train_label = batch['label']
        lanes_label = train_label[:,:, :144]
        lanes_label = self.label_formatting(lanes_label, is_get_label_as_tensor=False) # channel0 = line number, channel1 = confidence

        ls_lb_ext, ls_lb_cls = self.get_lane_exist_and_cls_wise_maps(lanes_label, is_one_hot=True, is_ret_list=True)
        EPS = 1e-12

        ### 1st Stage ###
        ext_loss = 0.
        cls_loss = 0.
        len_total_ext_row = 0.
        for idx_cls in range(self.num_cls):
            ext_loss += -torch.sum(ls_lb_ext[idx_cls]*torch.log(out[f'ext_{idx_cls}']+EPS))
            idx_ext = torch.where(ls_lb_ext[idx_cls][:,:,0]==1.)
            len_ext_row = len(idx_ext[1])
            len_total_ext_row += len_ext_row
            cls_loss += -torch.sum(ls_lb_cls[idx_cls][idx_ext]*torch.log(out[f'cls_{idx_cls}'][idx_ext]+EPS))
        
        ext_loss = ext_loss/(6.*144.)
        cls_loss = self.lambda_cls*cls_loss/len_total_ext_row
        ### 1st Stage ###

        ### 2nd Stage ###
        ext_loss2 = 0.
        cls_loss2 = 0.
        len_total_ext_row2 = 0.
        for idx_cls in range(self.num_cls):
            ext_loss2 += -torch.sum(ls_lb_ext[idx_cls]*torch.log(out[f'ext2_{idx_cls}']+EPS))
            idx_ext2 = torch.where(ls_lb_ext[idx_cls][:,:,0]==1.)
            len_ext_row2 = len(idx_ext2[1])
            len_total_ext_row2 += len_ext_row2
            cls_loss2 += -torch.sum(ls_lb_cls[idx_cls][idx_ext2]*torch.log(out[f'cls2_{idx_cls}'][idx_ext2]+EPS))
        
        ext_loss2 = ext_loss2/(6.*144.)
        cls_loss2 = self.lambda_cls*cls_loss2/len_total_ext_row2
        ### 2nd Stage ###

        loss = ext_loss + cls_loss + ext_loss2 + cls_loss2
        # print(f'ext_loss = {ext_loss}, cls_loss = {cls_loss}')

        return {'loss': loss, 'loss_stats': \
            {'ext_loss': ext_loss, 'cls_loss': cls_loss, 'ext_loss2': ext_loss2, 'cls_loss2': cls_loss2}}

    def get_lane_map_numpy_with_label(self, output, data, is_flip=True, is_img=False, is_get_1_stage_result=True):
        '''
        * in : output feature map
        * out: lane map with class or confidence
        *       per batch
        *       ### Label ###
        *       'conf_label': (144, 144) / 0, 1
        *       'cls_label': (144, 144) / 0, 1, 2, 3, 4, 5(lane), 255(ground)
        *       ### Raw Prediction ###
        *       'conf_pred_raw': (144, 144) / 0 ~ 1
        *       'cls_pred_raw': (7, 144, 144) / 0 ~ 1 (softmax)
        *       ### Confidence ###
        *       'conf_pred': (144, 144) / 0, 1 (thresholding)
        *       'conf_by_cls': (144, 144) / 0, 1 (only using cls)
        *       ### Classification ###
        *       'cls_idx': (144, 144) / 0, 1, 2, 3, 4, 5(lane), 255(ground)
        *       'conf_cls_idx': (144, 144) / (get cls idx in conf true positive)
        *       ### RGB Image ###
        *       'rgb_img_cls_label': (144, 144, 3)
        *       'rgb_img_cls_idx': (144, 144, 3)
        *       'rgb_img_conf_cls_idx': (144, 144, 3)
        '''
        lane_maps = dict()

        # for batch
        list_conf_label = []
        list_cls_label = []
        list_conf_pred_raw = []
        list_conf_pred = []
        list_cls_pred_raw = []
        list_cls_idx = []
        list_conf_by_cls = []
        list_conf_cls_idx = []

        batch_size = len(output['conf'])
        for batch_idx in range(batch_size):
            cls_label = data['label'][batch_idx].cpu().numpy()
            conf_label = np.where(cls_label == 255, 0, 1)

            conf_pred_raw = output['conf'][batch_idx].cpu().numpy()
            if is_flip:
                conf_pred_raw = np.flip(np.flip(conf_pred_raw, 0),1)
            conf_pred = np.where(conf_pred_raw > self.cfg.conf_thr, 1, 0)
            cls_pred_raw = torch.nn.functional.softmax(output['cls'][batch_idx], dim=0)
            cls_pred_raw = cls_pred_raw.cpu().numpy()
            if is_flip:
                cls_pred_raw = np.flip(np.flip(cls_pred_raw, 1),2)
            cls_idx = np.argmax(cls_pred_raw, axis=0)
            cls_idx[np.where(cls_idx==6)] = 255
            conf_by_cls = cls_idx.copy()
            conf_by_cls = np.where(conf_by_cls==255, 0, 1)
            conf_cls_idx = cls_idx.copy()
            conf_cls_idx[np.where(conf_pred==0)] = 255

            list_cls_label.append(cls_label)
            list_conf_label.append(conf_label)
            list_conf_pred_raw.append(conf_pred_raw)
            list_conf_pred.append(conf_pred)
            list_cls_pred_raw.append(cls_pred_raw)
            list_cls_idx.append(cls_idx)
            list_conf_by_cls.append(conf_by_cls)
            list_conf_cls_idx.append(conf_cls_idx)

        lane_maps.update({
            'conf_label': list_conf_label,
            'cls_label': list_cls_label,
            'conf_pred_raw': list_conf_pred_raw,
            'cls_pred_raw': list_cls_pred_raw,
            'conf_pred': list_conf_pred,
            'conf_by_cls': list_conf_by_cls,
            'cls_idx': list_cls_idx,
            'conf_cls_idx': list_conf_cls_idx,
        })

        if is_img:
            list_rgb_img_cls_label = []
            list_rgb_img_cls_idx = []
            list_rgb_img_conf_cls_idx = []

            for batch_idx in range(batch_size):
                list_rgb_img_cls_label.append(
                    self.get_rgb_img_from_cls_map(list_cls_label[batch_idx]))
                list_rgb_img_cls_idx.append(
                    self.get_rgb_img_from_cls_map(list_cls_idx[batch_idx]))
                list_rgb_img_conf_cls_idx.append(
                    self.get_rgb_img_from_cls_map(list_conf_cls_idx[batch_idx]))
            
            lane_maps.update({
                'rgb_cls_label': list_rgb_img_cls_label,
                'rgb_cls_idx': list_rgb_img_cls_idx,
                'rgb_conf_cls_idx': list_rgb_img_conf_cls_idx,
            })

            if is_get_1_stage_result:
                list_conf_label = []
                list_cls_label = []
                list_conf_pred_raw = []
                list_conf_pred = []
                list_cls_pred_raw = []
                list_cls_idx = []
                list_conf_by_cls = []
                list_conf_cls_idx = []

                list_rgb_img_conf_cls_idx = []

                batch_size = len(output['conf_1'])
                # print(batch_size)
                for batch_idx in range(batch_size):
                    cls_label = data['label'][batch_idx].cpu().numpy()
                    conf_label = np.where(cls_label == 255, 0, 1)

                    conf_pred_raw = output['conf_1'][batch_idx].cpu().numpy()
                    if is_flip:
                        conf_pred_raw = np.flip(np.flip(conf_pred_raw, 0),1)
                    conf_pred = np.where(conf_pred_raw > self.cfg.conf_thr, 1, 0)
                    cls_pred_raw = torch.nn.functional.softmax(output['cls_1'][batch_idx], dim=0)
                    cls_pred_raw = cls_pred_raw.cpu().numpy()
                    if is_flip:
                        cls_pred_raw = np.flip(np.flip(cls_pred_raw, 1),2)
                    cls_idx = np.argmax(cls_pred_raw, axis=0)
                    cls_idx[np.where(cls_idx==6)] = 255
                    conf_by_cls = cls_idx.copy()
                    conf_by_cls = np.where(conf_by_cls==255, 0, 1)
                    conf_cls_idx = cls_idx.copy()
                    conf_cls_idx[np.where(conf_pred==0)] = 255

                    list_cls_label.append(cls_label)
                    list_conf_label.append(conf_label)
                    list_conf_pred_raw.append(conf_pred_raw)
                    list_conf_pred.append(conf_pred)
                    list_cls_pred_raw.append(cls_pred_raw)
                    list_cls_idx.append(cls_idx)
                    list_conf_by_cls.append(conf_by_cls)
                    list_conf_cls_idx.append(conf_cls_idx)
                    list_rgb_img_conf_cls_idx.append(
                        self.get_rgb_img_from_cls_map(list_conf_cls_idx[batch_idx]))
                
                lane_maps.update({
                    'conf_pred_raw_1': list_conf_pred_raw,
                    'cls_pred_raw_1': list_cls_pred_raw,
                    'conf_pred_1': list_conf_pred,
                    'conf_by_cls_1': list_conf_by_cls,
                    'cls_idx_1': list_cls_idx,
                    'conf_cls_idx_1': list_conf_cls_idx,
                    'rgb_conf_cls_idx_1': list_rgb_img_conf_cls_idx,
                })

        # print(lane_maps.keys())

        return lane_maps

    def get_rgb_img_from_cls_map(self, cls_map):
        temp_rgb_img = np.zeros((144, 144, 3), dtype=np.uint8)
               
        for j in range(144):
            for i in range(144):
                idx_lane = int(cls_map[j,i])
                temp_rgb_img[j,i,:] = self.cfg.cls_lane_color[idx_lane] \
                                        if not (idx_lane == 255) else (0,0,0)

        return temp_rgb_img
        