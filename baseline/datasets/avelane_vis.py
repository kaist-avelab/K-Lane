'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import os.path as osp
import os
import numpy as np
import cv2
import torch
from glob import glob
import pickle

from torch.utils.data import Dataset

class AVELaneVis(Dataset):
    def __init__(self, data_root, split, description=None, cfg=None):
        self.cfg = cfg
        self.data_root = data_root
        self.training = 'train' in split
        self.num_seq = len(os.listdir(osp.join(data_root, 'train')))
        self.list_data_type = ['bev_image', 'bev_image_label', 'bev_tensor', 'frontal_img', 'pc']
        self.list_data_tail = ['.pickle', '.pickle', '.pickle', '.jpg', '.pcd']
        
        if split == 'train':
            self.data_infos = self.load_train_data_infos()
        elif split == 'test':
            self.data_infos = self.load_test_data_infos()

        if description:
            self.data_infos = self.filter_data_infos(description)

    def get_time_string(self, data_file):
        return data_file.split('.')[0].split('_')[-1]

    def load_train_data_infos(self):
        data_infos = []
        train_root = osp.join(self.data_root, 'train')
        for name_seq in os.listdir(train_root):
            # criterion: tensor_label
            list_tensor_label = sorted(os.listdir(osp.join(train_root, name_seq, 'bev_tensor_label')))
            list_list_data = list(map(lambda data_type: os.listdir(osp.join(train_root, name_seq, data_type)), self.list_data_type))
            temp_description = open(osp.join(train_root, name_seq, 'description.txt'), 'r')
            list_description = temp_description.readline()
            list_description = list_description.split(',')
            list_description[-1] = list_description[-1][:-1] # delete \n
            temp_description.close()

            temp_calib = open(osp.join(train_root, name_seq, 'calib_seq.txt'), 'r')
            val_calib = temp_calib.readline()
            val_calib = list(map(lambda x: float(x), val_calib.split(',')[:-1]))
            temp_calib.close()

            for name_tensor_label in list_tensor_label:
                temp_data_info = dict()
                temp_data_info['bev_tensor_label'] = osp.join(train_root, name_seq, 'bev_tensor_label', name_tensor_label)
                temp_data_info['description'] = list_description
                time_string = self.get_time_string(name_tensor_label)
                temp_data_info['timestamp'] = time_string
                temp_data_info['seq'] = name_seq
                temp_data_info['calib'] = val_calib
                
                for idx, data_type in enumerate(self.list_data_type):
                    temp_data_name = data_type + '_' + time_string + self.list_data_tail[idx]
                    if temp_data_name in list_list_data[idx]:
                        temp_data_info[data_type] = osp.join(train_root, name_seq, data_type, temp_data_name)
                    else:
                        temp_data_info[data_type] = None
                data_infos.append(temp_data_info)

        data_infos.sort(key = lambda x: x['seq'])
        
        return data_infos

    def get_desc_test_per_frame(self):
        path_description_test = osp.join(self.data_root, 'description_frames_test.txt')
        f = open(path_description_test, 'r')
        lines = f.readlines()
        f.close()

        lines = list(map(lambda line: line.split(','), lines))
        
        dict_num_lanes = dict()
        dict_dsec = dict()
        for i in range(len(lines)):
            if not (i == (len(lines)-1)):
                lines[i][-1] = lines[i][-1][:-1] # get rid of \n
            name_key = lines[i][0]
            num_lane = int(lines[i][1])
            dict_num_lanes.update({name_key: num_lane})
            temp_list_desc = lines[i][2:].copy()
            temp_list_desc = list(map(lambda x: x[1:], temp_list_desc))
            # print(temp_list_desc)
            dict_dsec.update({name_key: temp_list_desc})

        return dict_dsec, dict_num_lanes

    def load_test_data_infos(self):
        data_infos = []
        train_root = osp.join(self.data_root, 'train')
        test_root = osp.join(self.data_root, 'test')
        
        dict_dsec, dict_num_lanes = self.get_desc_test_per_frame()

        dict_calib = dict()

        # print(test_root)
        
        list_time_string = []
        list_corresponding_seq = []
        list_list_description = []
        list_list_calib = []
        list_list_data = [[],[],[],[],[]]
        # time_string, seq
        for name_seq in os.listdir(train_root):
            # criterion: 'bev_tensor'
            temp_list_time_string = list(map(self.get_time_string, sorted(os.listdir(osp.join(train_root, name_seq, 'bev_tensor')))))
            list_time_string.extend(temp_list_time_string)
            temp_list_corresponding_seq = [name_seq]*len(temp_list_time_string)
            list_corresponding_seq.extend(temp_list_corresponding_seq)
            
            temp_description = open(osp.join(train_root, name_seq, 'description.txt'), 'r')
            list_description = temp_description.readline()
            list_description = list_description.split(',')
            list_description[-1] = list_description[-1][:-1] # delete \n
            temp_description.close()
            list_list_description.append(list_description)

            temp_calib = open(osp.join(train_root, name_seq, 'calib_seq.txt'), 'r')
            val_calib = temp_calib.readline()
            val_calib = list(map(lambda x: float(x), val_calib.split(',')[:-1]))
            temp_calib.close()
            # list_list_calib.append(val_calib)
            dict_calib[name_seq] = val_calib
            
            temp_list_list_data = list(map(lambda data_type: os.listdir(osp.join(train_root, name_seq, data_type)), self.list_data_type))
            for i in range(len(temp_list_list_data)):
                list_list_data[i].extend(temp_list_list_data[i])
        # print(list_list_data)

        for name_tensor_label in sorted(os.listdir(test_root)):
            temp_data_info = dict()
            time_string = self.get_time_string(name_tensor_label)
            temp_data_info['timestamp'] = time_string
            
            corresponding_idx = list_time_string.index(time_string)

            temp_data_info['bev_tensor_label'] = osp.join(test_root, name_tensor_label)
            name_seq = list_corresponding_seq[corresponding_idx]
            temp_data_info['seq'] = name_seq
            list_temp_description = list_list_description[int(name_seq.split('_')[-1])-1].copy()
            # list_temp_description.extend(dict_dsec[time_string])
            # print(list_temp_description)
            # temp_data_info['description'] = list_temp_description
            # temp_data_info['num_lanes'] = dict_num_lanes[time_string]
            
            # temp_data_info['calib'] = list_list_calib[int(name_seq.split('_')[-1])-1]
            temp_data_info['calib'] = dict_calib[name_seq]

            for idx, data_type in enumerate(self.list_data_type):
                temp_data_name = data_type + '_' + time_string + self.list_data_tail[idx]
                # print(temp_data_name)
                if temp_data_name in list_list_data[idx]:
                    temp_data_info[data_type] = osp.join(train_root, name_seq, data_type, temp_data_name)
                else:
                    temp_data_info[data_type] = None
            data_infos.append(temp_data_info)

        data_infos.sort(key=lambda x: x['seq'])

        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        
        if not osp.isfile(data_info['bev_tensor']):
            raise FileNotFoundError('cannot find file: {}'.format(data_info['bev_tensor']))

        if not osp.isfile(data_info['bev_tensor_label']):
            raise FileNotFoundError('cannot find file: {}'.format(data_info['bev_tensor_label']))

        meta = data_info.copy()
        sample = dict()
        sample['meta'] = meta

        with open(meta['bev_tensor'], 'rb') as f:
            bev_tensor = pickle.load(f, encoding='latin1')
        with open(meta['bev_tensor_label'], 'rb') as f:
            bev_tensor_label = pickle.load(f, encoding='latin1')

        pillars = np.squeeze(bev_tensor[0], axis=0)
        pillar_indices = np.squeeze(bev_tensor[1], axis=0)

        sample['pillars'] = pillars
        sample['pillar_indices'] = pillar_indices
        sample['label'] = bev_tensor_label[:,0:144]
        sample['rowise_existence'] = bev_tensor_label[:,144:]

        return sample

if __name__ == '__main__':
    AVELaneVis()
        