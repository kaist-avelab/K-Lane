'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import torch
import torch.nn as nn
import numpy as np

from baseline.models.registry import HEADS

@HEADS.register_module
class GridSeg(nn.Module):
    
    def __init__(self,
                num_1=1024,
                num_2=2048,
                num_classes=7,
                cfg=None):
        super(GridSeg, self).__init__()
        self.cfg=cfg
        
        self.act_sigmoid = nn.Sigmoid()

        self.conf_predictor = nn.Sequential(
            nn.Conv2d(num_1, num_2, 1),
            nn.Conv2d(num_2, 1, 1)
        )

        self.class_predictor = nn.Sequential(
            nn.Conv2d(num_1, num_2, 1),
            nn.Conv2d(num_2, num_classes, 1)
        )

    def forward(self, x):
        conf_output = self.act_sigmoid(self.conf_predictor(x))
        class_output = self.class_predictor(x)

        out = torch.cat((class_output, conf_output), 1)
        
        return out

    def label_formatting(self, raw_label):
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
                    # y_idx = i
                    # x_idx = j

                    line_num = int(label_data[i][j])
                    if line_num == 255:
                        label_temp[y_idx][x_idx][1] = 0
                        # classification
                        label_temp[y_idx][x_idx][0] = 6
                    else: # 클래스
                        # confidence
                        label_temp[y_idx][x_idx][1] = 1
                        # classification
                        label_temp[y_idx][x_idx][0] = line_num

            label_tensor[k,:,:,:] = np.transpose(label_temp, (2, 0, 1))

        return(torch.tensor(label_tensor))
    
    def loss(self, out, batch):
        train_label = batch['label']
        lanes_label = train_label[:,:, :144]
        lanes_label = self.label_formatting(lanes_label) #channel0 = line number, channel1 = confidence

        y_pred_cls = out[:, 0:7, :, :]
        y_pred_conf = out[:, 7, :, :]

        y_label_cls = lanes_label[:, 0, :, :].cuda()
        y_label_conf = lanes_label[:, 1, :, :].cuda()

        cls_loss = 0
        cls_loss += nn.CrossEntropyLoss()(y_pred_cls, y_label_cls)

        ## Dice Loss ###
        numerator = 2 * torch.sum(torch.mul(y_pred_conf, y_label_conf))
        denominator = torch.sum(torch.square(y_pred_conf)) + torch.sum(torch.square(y_label_conf)) + 1e-6
        dice_coeff = numerator / denominator
        conf_loss = (1 - dice_coeff)

        loss = conf_loss + cls_loss

        ret = {'loss': loss, 'loss_stats': {'conf': conf_loss, 'cls': cls_loss}}

        return ret

    def get_lane_map_numpy_with_label(self, output, data, is_flip=True, is_img=False):
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

        return lane_maps

    def get_rgb_img_from_cls_map(self, cls_map):
        temp_rgb_img = np.zeros((144, 144, 3), dtype=np.uint8)
               
        for j in range(144):
            for i in range(144):
                idx_lane = int(cls_map[j,i])
                temp_rgb_img[j,i,:] = self.cfg.cls_lane_color[idx_lane] \
                                        if not (idx_lane == 255) else (0,0,0)

        return temp_rgb_img
        