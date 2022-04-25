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

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

@HEADS.register_module
class RowSharNotReducRef(nn.Module):
    def __init__(self,
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
                cfg=None):
        super(RowSharNotReducRef, self).__init__()
        self.cfg=cfg

        ### Making Labels ###
        self.num_cls = 6
        self.lambda_cls=lambda_cls
        ### Making Labels ###

        ### MLP Encoder (1st Stage) ###
        self.row_tensor_maker = Rearrange('b c h w -> b (c w) h')

        for idx_cls in range(self.num_cls):
            setattr(self, f'ext_{idx_cls}', nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,2,1,1,0),
                Rearrange('b c h -> b h c')
            ))

            setattr(self, f'cls_{idx_cls}', nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,row_size,1,1,0),
                Rearrange('b w h -> b h w')
            ))
        ### MLP Encoder (1st Stage) ###

        ### Refinement (2nd Stage) ###
        self.thr_ext = thr_ext
        self.off_grid = off_grid
        in_token_channel = (2*self.off_grid+1)*row_size*dim_feat
        self.to_token = nn.Sequential(
            Rearrange('c h w -> (c h w)'),
            nn.Linear(in_token_channel, dim_token)
        )
        for idx_cls in range(self.num_cls):
            setattr(self, f'emb_{idx_cls}', nn.Parameter(torch.randn(dim_token)).cuda())
        self.emb_dropout = None
        if tr_emb_dropout != 0.:
            self.emb_dropout = nn.Dropout(tr_emb_dropout)
        self.tr_lane_correlator = nn.Sequential(
            Transformer(dim_token, tr_depth, tr_heads, tr_dim_head, tr_mlp_dim, tr_dropout),
            nn.LayerNorm(dim_token),
            nn.Linear(dim_token, in_token_channel),
            Rearrange('b n (c h w) -> b n c h w', c=dim_feat, h=row_size)
        )

        if not is_reuse_same_network:
            self.is_reuse_same_network = False
            for idx_cls in range(self.num_cls):
                setattr(self, f'ext2_{idx_cls}', nn.Sequential(
                    nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                    nn.BatchNorm1d(dim_shared),
                    nn.Conv1d(dim_shared,2,1,1,0),
                    Rearrange('b c h -> b h c')
                ))

                setattr(self, f'cls2_{idx_cls}', nn.Sequential(
                    nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                    nn.BatchNorm1d(dim_shared),
                    nn.Conv1d(dim_shared,row_size,1,1,0),
                    Rearrange('b w h -> b h w')
                ))
        else:
            is_reuse_same_network = True

    def forward(self, x):
        out_dict = dict()
        b, _, _, _ = x.shape
        self.b_size = b

        row_feat = x # self.botn_layer(x)
        row_tensor = self.row_tensor_maker(row_feat)
        
        for idx_cls in range(self.num_cls):
            out_dict.update({f'ext_{idx_cls}': torch.nn.functional.softmax(getattr(self, f'ext_{idx_cls}')(row_tensor),dim=2)})
            out_dict.update({f'cls_{idx_cls}': torch.nn.functional.softmax(getattr(self, f'cls_{idx_cls}')(row_tensor),dim=2)})

        ### 1st Stage Processing ###
        # b, 144, 144 conf
        # b, 7, 144, 144 cls
        b_size, dim_feat, img_h, img_w = row_feat.shape # 2, 16, 144, 144
        num_cls = self.num_cls

        # Zero padding for offset
        off_grid = self.off_grid
        zero_left = torch.zeros((b_size,dim_feat,img_h,off_grid)).cuda()
        zero_right = torch.zeros((b_size,dim_feat,img_h,off_grid)).cuda()
        row_feat_pad = torch.cat([zero_left, row_feat, zero_right], dim=3) # 2, 16, 144, 148 (144+2*off_grid)
        # print(row_feat_pad.shape)

        for idx_b in range(b_size):
            ext_lane_idx_per_b = []
            ext_lane_tokens = []
            ext_corr_idxs = []
            for idx_c in range(num_cls):
                # exist: one-hot to 0 (line), 1 (not)
                lane_ext_prob = torch.mean(out_dict[f'ext_{idx_c}'][idx_b,:,0]) # prob that is lane
                if lane_ext_prob > self.thr_ext: # if the line exts
                    ext_lane_idx_per_b.append(idx_c)
                    corr_idxs_b4 = out_dict[f'cls_{idx_c}'][idx_b,:,:] # when debug, make it small e.g., [idx_b,:8,:]
                    corr_idxs = torch.argmax(corr_idxs_b4, dim=1) # 144, 1 (batch 1)
                    ext_corr_idxs.append(corr_idxs)
                    
                    temp_token = torch.zeros((dim_feat,img_h,1+2*off_grid)).cuda()
                    for idx_h in range(img_h):
                        corr_idx = corr_idxs[idx_h]+off_grid # do not forget off_grid
                        # print(row_feat_pad[idx_b,:,idx_h,corr_idx-off_grid:corr_idx+off_grid].shape)
                        temp_token[:,idx_h,:] = row_feat_pad[idx_b,:,idx_h,corr_idx-off_grid:corr_idx+off_grid+1]
                    # linear transform & pos
                    # print(temp_token.shape)
                    temp_token = self.to_token(temp_token) + getattr(self, f'emb_{idx_c}')
                    ext_lane_tokens.append(torch.unsqueeze(torch.unsqueeze(temp_token, 0),0))

            # print(ext_lane_idx_per_b)
            # print(len(ext_lane_tokens))
            if len(ext_lane_tokens) > 0:
                tokens = self.tr_lane_correlator(torch.cat(ext_lane_tokens, dim=1))
                # print(tokens.shape)
                
                # return to original row_feat_pad
                for idx, corr_idxs in enumerate(ext_corr_idxs):
                    for idx_h in range(idx_h):
                        corr_idx = corr_idxs[idx_h]+off_grid
                        row_feat_pad[idx_b,:,idx_h,corr_idx-off_grid:corr_idx+off_grid+1] = tokens[0,idx,:,idx_h,:]
        row_feat = row_feat_pad[:,:,:,off_grid:img_w+off_grid]
        # print(row_feat.shape)
        row_tensor = self.row_tensor_maker(row_feat)
        ### 1st Stage Processing ###

        ### 2nd Stage ###
        for idx_cls in range(self.num_cls):
            if self.is_reuse_same_network:
                out_dict.update({f'ext2_{idx_cls}': torch.nn.functional.softmax(getattr(self, f'ext_{idx_cls}')(row_tensor),dim=2)})
                out_dict.update({f'cls2_{idx_cls}': torch.nn.functional.softmax(getattr(self, f'cls_{idx_cls}')(row_tensor),dim=2)})
            else:
                out_dict.update({f'ext2_{idx_cls}': torch.nn.functional.softmax(getattr(self, f'ext2_{idx_cls}')(row_tensor),dim=2)})
                out_dict.update({f'cls2_{idx_cls}': torch.nn.functional.softmax(getattr(self, f'cls2_{idx_cls}')(row_tensor),dim=2)})
        ### 2nd Stage ###

        return out_dict

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
        