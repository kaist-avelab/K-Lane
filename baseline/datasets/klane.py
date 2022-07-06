'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import os.path as osp
import os
from re import M
import numpy as np
import cv2
import torch
from glob import glob
import pickle
import open3d as o3d
from torch.utils.data import Dataset

try:
    from baseline.utils.pc_utils import *
    from baseline.datasets.registry import DATASETS
except:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from baseline.utils.pc_utils import *
    from baseline.datasets.registry import DATASETS

@DATASETS.register_module
class KLane(Dataset):
    def __init__(self, data_root, split, mode_item='pc', description=None, cfg=None):
        self.cfg = cfg
        self.data_root = data_root
        self.mode_item = mode_item
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

            for name_tensor_label in list_tensor_label:
                temp_data_info = dict()
                temp_data_info['bev_tensor_label'] = osp.join(train_root, name_seq, 'bev_tensor_label', name_tensor_label)
                temp_data_info['description'] = list_description
                time_string = self.get_time_string(name_tensor_label)
                
                for idx, data_type in enumerate(self.list_data_type):
                    temp_data_name = data_type + '_' + time_string + self.list_data_tail[idx]
                    if temp_data_name in list_list_data[idx]:
                        temp_data_info[data_type] = osp.join(train_root, name_seq, data_type, temp_data_name)
                    else:
                        temp_data_info[data_type] = None
                data_infos.append(temp_data_info)
        return data_infos

    def load_test_data_infos(self):
        data_infos = []
        train_root = osp.join(self.data_root, 'train')
        test_root = osp.join(self.data_root, 'test')
        
        test_descriptions_path = osp.join(self.data_root, 'description_frames_test.txt')
        list_test_descriptions = []
        with open(test_descriptions_path, 'r') as f:
            for line in f:
                list_test_descriptions.append(line.strip('\n').split(', '))

        list_time_string = []
        list_corresponding_seq = []
        list_list_description = []
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
            
            temp_list_list_data = list(map(lambda data_type: os.listdir(osp.join(train_root, name_seq, data_type)), self.list_data_type))
            for i in range(len(temp_list_list_data)):
                list_list_data[i].extend(temp_list_list_data[i])
        # print(list_list_data)

        for name_tensor_label in sorted(os.listdir(test_root)):
            temp_data_info = dict()
            time_string = self.get_time_string(name_tensor_label)
            
            corresponding_idx = list_time_string.index(time_string)

            temp_data_info['bev_tensor_label'] = osp.join(test_root, name_tensor_label)
            name_seq = list_corresponding_seq[corresponding_idx]
            # temp_data_info['description'] = list_list_description[int(name_seq.split('_')[-1])-1]
            
            for desc in list_test_descriptions:
                if desc[0] == time_string:
                    if self.cfg.is_eval_conditional:
                        temp_data_info['description'] = desc
                    else:
                        temp_data_info['description'] = desc[-3:] # modified by Xiaoxin
                    break

            for idx, data_type in enumerate(self.list_data_type):
                temp_data_name = data_type + '_' + time_string + self.list_data_tail[idx]
                # print(temp_data_name)
                if temp_data_name in list_list_data[idx]:
                    temp_data_info[data_type] = osp.join(train_root, name_seq, data_type, temp_data_name)
                else:
                    temp_data_info[data_type] = None
            data_infos.append(temp_data_info)
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.mode_item == 'pillar':
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

        elif self.mode_item == 'pc':
            data_info = self.data_infos[idx]

            if not osp.isfile(data_info['pc']):
                raise FileNotFoundError('cannot find file: {}'.format(data_info['pc']))

            if not osp.isfile(data_info['bev_tensor_label']):
                raise FileNotFoundError('cannot find file: {}'.format(data_info['bev_tensor_label']))

            meta = data_info.copy()
            sample = dict()
            sample['meta'] = meta

            with open(meta['bev_tensor_label'], 'rb') as f:
                bev_tensor_label = pickle.load(f, encoding='latin1')
            sample['label'] = bev_tensor_label[:,0:144]

            pc = get_pc_os64_with_path(meta['pc'])
            pc = filter_pc_os64_with_roi(pc, self.cfg.list_filter_roi, self.cfg.filter_mode)
            pc = append_image_index_to_pc_os64(pc, self.cfg.list_roi_xy, self.cfg.list_grid_xy)

            sample['proj'] = get_projection_image_from_pointclouds(pc, list_img_size_xy=self.cfg.list_img_size_xy, is_flip=False)
            sample['proj'] = np.transpose(sample['proj'], (2,0,1))
            sample['proj'] = sample['proj'].astype(np.float32)
            # print(sample['proj'].shape)

        elif self.mode_item == 'pc_div':
            data_info = self.data_infos[idx]

            if not osp.isfile(data_info['pc']):
                raise FileNotFoundError('cannot find file: {}'.format(data_info['pc']))

            if not osp.isfile(data_info['bev_tensor_label']):
                raise FileNotFoundError('cannot find file: {}'.format(data_info['bev_tensor_label']))

            meta = data_info.copy()
            sample = dict()
            sample['meta'] = meta

            with open(meta['bev_tensor_label'], 'rb') as f:
                bev_tensor_label = pickle.load(f, encoding='latin1')
            sample['label'] = bev_tensor_label[:,0:144]

            pc = get_pc_os64_with_path(meta['pc'])

            pc_up = filter_pc_os64_with_roi(pc, self.cfg.list_filter_roi_up, self.cfg.filter_mode_up)
            pc_up = append_image_index_to_pc_os64(pc, self.cfg.list_roi_xy_up, self.cfg.list_grid_xy_up)
            
            pc_do = filter_pc_os64_with_roi(pc, self.cfg.list_filter_roi_do, self.cfg.filter_mode_do)
            pc_do = append_image_index_to_pc_os64(pc, self.cfg.list_roi_xy_do, self.cfg.list_grid_xy_do)

            sample['proj_up'] = get_projection_image_from_pointclouds(pc_up, is_flip=False)
            sample['proj_up'] = np.transpose(sample['proj_up'], (2,0,1))
            sample['proj_up'] = sample['proj_up'].astype(np.float32)

            sample['proj_do'] = get_projection_image_from_pointclouds(pc_do, is_flip=False)
            sample['proj_do'] = np.transpose(sample['proj_do'], (2,0,1))
            sample['proj_do'] = sample['proj_do'].astype(np.float32)

        return sample

def checking_none_data(dataset_path='/media/donghee/HDD_0/KLane', dataset_type='train'):
    dataset = KLane(dataset_path, dataset_type)

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print(f'* full data length = {len(loader)}')
    print(f'* iterate until {len(loader)}, plz wait')

    cnt_batch_data_none_counter = [0, 0, 0, 0] # 'pillars', 'pillar_indices', 'label', 'rowise_existence'
    cnt_meta_data_none_counter = [0, 0, 0, 0, 0, 0, 0] # 'bev_tensor_label', 'description', 'bev_image', 'bev_image_label', 'bev_tensor', 'frontal_img', 'pc'

    from tqdm import tqdm
    for batch_idx, batch_data in tqdm(enumerate(loader)):
        idx_cnt = 0
        for k, v in batch_data.items():
            if k == 'meta':
                continue
            if v is None:
                cnt_batch_data_none_counter[idx_cnt] += 1
            idx_cnt += 1

        idx_cnt = 0
        for k, v in batch_data['meta'].items():
            if None in v:
                cnt_meta_data_none_counter[idx_cnt] += 1
            idx_cnt += 1

    print(cnt_batch_data_none_counter)
    print(cnt_meta_data_none_counter)

def visualize_pc_data(dataset_path='/media/donghee/HDD_0/KLane', dataset_type='train'):
    dataset = KLane(dataset_path, dataset_type)

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print(f'* full data length = {len(loader)}')
    print(f'* iterate until {len(loader)}, plz wait')

    # cnt_batch_data_none_counter = [0, 0, 0, 0] # 'pillars', 'pillar_indices', 'label', 'rowise_existence'
    # cnt_meta_data_none_counter = [0, 0, 0, 0, 0, 0, 0] # 'bev_tensor_label', 'description', 'bev_image', 'bev_image_label', 'bev_tensor', 'frontal_img', 'pc'

    from tqdm import tqdm
    for batch_idx, batch_data in tqdm(enumerate(loader)):
        path_pcd = batch_data['meta']['pc'][0]
        print(path_pcd)

        current_point_cloud = o3d.io.read_point_cloud(path_pcd)

        print(dir(current_point_cloud))
        
        o3d.visualization.draw_geometries([current_point_cloud])

        import sys
        sys.exit()

def visualize_pc_data_with_roi(dataset_path='/media/donghee/HDD_0/KLane'):
    # list_roi_xyz=[0.02, 46.08, -11.52, 11.52, -2.0, 1.5]
    list_filter_roi=[0.02, 46.08, -11.52, 11.52, -2.0, -1.2] # do
    # list_filter_roi=[0.02, 46.08, -11.52, 11.52, -1.3, 1.5]  # up
    list_roi_xy = [0.0, 46.08, -11.52, 11.52]
    list_grid_xy = [0.04, 0.02]
    list_img_size_xy = [1152, 1152]
    print(dataset_path)

    klane = KLane(dataset_path, 'train', 'pc')
    data_idx = 900
    
    ### Meta data ###
    print(klane.data_infos[0].keys())

    path_pcd = klane.data_infos[data_idx]['pc']
    pc_os64 = get_pc_os64_with_path(path_pcd)

    x_min, x_max, y_min, y_max, z_min, z_max = list_filter_roi
    list_pc_values = pc_os64['values'].tolist()
    pc_os64['values'] = np.array(list(filter(lambda point: \
        (point[0] > x_min) and (point[0] < x_max) and \
        (point[1] > y_min) and (point[1] < y_max) and \
        (point[2] > z_min) and (point[2] < z_max), list_pc_values)))

    pc_os64 = append_image_index_to_pc_os64(pc_os64, list_roi_xy, list_grid_xy)
    
    img = get_projection_image_from_pointclouds(pc_os64, is_flip=False)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    # print(pc_os64['values'].shape)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.array(pc_os64['values'][:,:3]))
    # o3d.visualization.draw_geometries([pcd])

if __name__=='__main__':
    ### Checking None Data ###
    # checking_none_data(dataset_type='test')
    ### Checking None Data ###

    visualize_pc_data_with_roi()

    # visualize_pc_data()
    # from baseline.utils.config import Config
    # cfg = Config.fromfile('./configs/klane_mixer/projection_resnet50.py')
    # dataset = KLane('/media/donghee/HDD_0/KLane', 'train', cfg=cfg)
    # # dataset = AVELane('/media/donghee/HDD_0/KLane', 'test')

    # from torch.utils.data import DataLoader
    # loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    # print(len(loader))

    # from tqdm import tqdm
    # for batch_idx, batch_data in enumerate(loader):

    #     # print(batch_data)
        
    #     import cv2
    #     img = batch_data['proj'][0,1,:,:].detach().cpu().numpy()
    #     # img = np.transpose(img, (1, 2, 0))
    #     # img[np.where(img>0.5)] = 1
    #     print(img.shape)
    #     cv2.imwrite('./input_img.png', img*255)
    #     cv2.imwrite('./label.png', batch_data['label'][0].numpy().astype(np.uint8))
    #     # cv2.waitKey(0)

    #     import sys
    #     sys.exit()
