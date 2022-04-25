'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import os
import sys
import pickle
import numpy as np
import cv2
import torch
import open3d as o3d

from baseline.datasets.avelane_vis import AVELaneVis
from PyQt5.QtWidgets import QDesktopWidget, QListWidgetItem
from PyQt5.QtGui import QImage, QPixmap

from baseline.utils.config import Config
from baseline.engine.runner import Runner
from baseline.utils.pc_utils import *
import configs.config_vis as cnf

__all__ = ['move_window_to_center', 'load_dataset_vis', 'load_config_and_runner', \
            'get_rgb_img_from_path_label', 'get_q_pixmap_from_cv_img', \
            'get_list_p_text_edit', 'get_intrinsic_and_extrinsic_params_from_text_edit', \
            'get_rotation_and_translation_from_extrinsic', 'get_point_cloud_from_bev_tensor_label', \
            'get_pointcloud_with_rotation_and_translation', 'get_pixel_from_point_cloud_in_camera_coordinate', \
            'get_camera_img_as_connected_line_with_projected_pixel', 'get_output_from_runner', \
            'get_camera_img_as_dot_with_projected_pixel', 'get_metrics_conf_cls', \
            'get_o3d_pointcloud', 'get_wide_bev_map_with_white_background_for_vis', \
            'get_corresponding_sample_from_dataset_type', 'get_white_background_label', \
            'get_projected_img_from_pre_calib', ]

def move_window_to_center(p_mf):
    qr = p_mf.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    p_mf.move(qr.topLeft())

def load_dataset_vis(data_root, split, p_list_widget=None):
    dataset = AVELaneVis(data_root, split)

    if p_list_widget is None:
        return dataset

    p_list_widget.clear()
    for i in range(len(dataset.data_infos)):
        data_info = dataset.data_infos[i]

        temp_item = QListWidgetItem()
        temp_item.setData(1, data_info)
        temp_item.setText(str(i) + '. ' + data_info['seq'] + ', ' + data_info['timestamp'])
        p_list_widget.addItem(temp_item)

    return dataset

def load_config_and_runner(path_config, gpus):
    cfg = Config.fromfile(path_config)
    cfg.log_dir = cfg.log_dir + '/vis'
    os.makedirs(cfg.log_dir, exist_ok=True)
    cfg.work_dirs = cfg.log_dir + '/' + cfg.dataset.train.type
    cfg.gpus = len(gpus.split(','))
    runner = Runner(cfg)

    return cfg, runner

def get_intrinsic_and_extrinsic_params_from_text_edit(list_p_text_edit):
    '''
    * return
    *   1. intrinsic (fx, fy, px, py)
    *   2. extrinsic (roll, pitch, yaw, x, y, z) [degree, m]
    '''
    assert len(list_p_text_edit) == 4, 'get 4 rows'

    list_rows = list(map(lambda text_edit: text_edit.toPlainText(), list_p_text_edit))
    list_list_params = []
    temp_list_params = list(map(lambda plain_text: plain_text.split(','), list_rows))
    for list_params in temp_list_params:
        list_params = list(map(lambda param: float(param), list_params))
        list_list_params.append(list_params)
    
    intrinsic = [list_list_params[0][0], list_list_params[1][1], list_list_params[0][2], list_list_params[1][2]]
    extrinsic = []
    extrinsic.extend(list_list_params[2])
    extrinsic.extend(list_list_params[3])

    return intrinsic, extrinsic

def get_list_p_text_edit(p_mf):
    list_p_text_edit = []
    
    for i in range(4):
        list_p_text_edit.append(getattr(p_mf, 'plainTextEdit_row_'+str(i+1)))
    
    return list_p_text_edit

def get_rgb_img_from_path_label(path_bev_tensor_label):
    with open(path_bev_tensor_label, 'rb') as f:
            bev_tensor_label = pickle.load(f, encoding='latin1')
    bev_tensor_label = bev_tensor_label[:,0:144]

    bev_tensor_label_img = np.zeros_like(bev_tensor_label, dtype=np.uint8)
    bev_tensor_label_img = cv2.cvtColor(bev_tensor_label_img, cv2.COLOR_GRAY2RGB)
    
    for i in range(6):
        bev_tensor_label_img[np.where(bev_tensor_label==i)] = cnf.cls_lane_color[i]
    
    return bev_tensor_label_img

def cv_img_to_q_image(cv_img):
    if len(np.shape(cv_img))==2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)

    height, width, _ = cv_img.shape

    bytes_per_line = 3 * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    q_img = q_img.rgbSwapped()

    return q_img

def get_q_pixmap_from_cv_img(cv_img, width=None, height=None, interpolation=cv2.INTER_LINEAR):
    if width and height:
        cv_img = cv2.resize(cv_img, dsize=(width, height), interpolation=interpolation)
    
    return QPixmap.fromImage(cv_img_to_q_image(cv_img))

def get_rotation_and_translation_from_extrinsic(extrinsic, is_deg = True):
    ext_copy = extrinsic.copy() # if not copy, will change the parameters permanently
    if is_deg:
        ext_copy[:3] = list(map(lambda x: x*np.pi/180., extrinsic[:3]))

    roll, pitch, yaw = ext_copy[:3]
    x, y, z = ext_copy[3:]

    ### Roll-Pitch-Yaw Convention
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)
    c_r = np.cos(roll)
    s_r = np.sin(roll)

    R_yaw = np.array([[c_y, -s_y, 0.],[s_y, c_y, 0.],[0., 0., 1.]])
    R_pitch = np.array([[c_p, 0., s_p],[0., 1., 0.],[-s_p, 0., c_p]])
    R_roll = np.array([[1., 0., 0.],[0., c_r, -s_r],[0., s_r, c_r]])

    R = np.dot(np.dot(R_yaw, R_pitch), R_roll)
    trans = np.array([[x],[y],[z]])

    return R, trans

def get_point_cloud_from_bev_tensor_label(bev_label, with_cls=False, z_fix=cnf.z_fix):
    '''
    * return
    *   n x 3 (x,y,z) [m] in np.array, with_cls == False
    *   n x 3 (x,y,z) [m], n x 1 (cls_idx) in np.array, with_cls == True
    '''
    bev_label_144 = bev_label[:,:144]

    points_arr = []
    cls_arr = []
    for i in range(6):
        points_in_pixel = np.where(bev_label_144==i)
        _, num_points = np.shape(points_in_pixel)
        for j in range(num_points):
            x_point, y_point = get_point_from_pixel_in_m(points_in_pixel[1][j], points_in_pixel[0][j])
            points_arr.append([x_point, y_point, z_fix])
            if with_cls:
                cls_arr.append(i) # cls
    
    if with_cls:
        return np.array(points_arr), np.array(cls_arr)
    else:
        return np.array(points_arr)

def get_point_from_pixel_in_m(x_pix, y_pix):
    x_lidar = 144 - (y_pix+0.5)
    y_lidar = 72 - (x_pix+0.5)
    
    return cnf.x_grid*x_lidar, cnf.y_grid*y_lidar

def get_pointcloud_with_rotation_and_translation(point_cloud_xyz, rot, tra):
    pc_xyz = point_cloud_xyz.copy()
    num_points = len(pc_xyz)
    
    for i in range(num_points):
        point_temp = pc_xyz[i,:]
        point_temp = np.reshape(point_temp, (3,1))

        point_processed = np.dot(rot, point_temp) + tra
        point_processed = np.reshape(point_processed, (3,))
        
        pc_xyz[i,:] = point_processed

    return pc_xyz

def get_pixel_from_point_cloud_in_camera_coordinate(point_cloud_xyz, intrinsic):
    '''
    * in : pointcloud in np array (nx3)
    * out: projected pixel in np array (nx2)
    '''

    process_pc = point_cloud_xyz.copy()
    if (np.shape(point_cloud_xyz) == 1):
        num_points = 0
    else:
        #Temporary fix for when shape = (0.)
        try:
            num_points, _ = np.shape(point_cloud_xyz)
        except:
            num_points = 0
    fx, fy, px, py = intrinsic

    pixels = []
    for i in range(num_points):
        xc, yc, zc = process_pc[i,:]
        y_pix = py - fy*zc/xc
        x_pix = px - fx*yc/xc

        pixels.append([x_pix, y_pix])
    pixels = np.array(pixels)

    return pixels

def get_camera_img_as_dot_with_projected_pixel(img_cam, pixels, with_label=False):
    height, width, _ = np.shape(img_cam)

    # print(pixels)
    # print(np.shape(pixels))
    pixels = np.array(list(filter(lambda x: (x[0]>=0) and (x[0]<width-0.5) and \
                                            (x[1]>=0) and (x[1]<height-0.5), pixels)))
    if with_label:
        process_pixel = pixels[:,:2]
        process_label = pixels[:,2:]
    else:
        process_pixel = pixels

    # round digits
    process_pixel = np.around(process_pixel)
    num_points, _ = np.shape(process_pixel)

    for i in range(num_points):
        x_pix, y_pix = process_pixel[i,:].astype(int)
        if with_label:
            color = cnf.LINE_COLOR_LIST[int(process_label[i])]
        else:
            color = (255, 180, 0)
        img_cam = cv2.line(img_cam, (x_pix, y_pix), (x_pix, y_pix), color, cnf.THICKNESS)

    return img_cam

def get_camera_img_as_dot_with_projected_pixel(img_cam, pixels, arr_cls_idx=None, line_width=None):
    height, width, _ = np.shape(img_cam)

    process_pixel = pixels.copy()
    if arr_cls_idx is not None:
        if len(np.shape(pixels))==1:
            return img_cam
        num_points, _ = np.shape(pixels)
        process_cls_idx = np.reshape(arr_cls_idx.copy(),(num_points,1))
        process_pixel = np.concatenate((process_pixel, process_cls_idx), axis=1)
    temp_process_pixel = np.array(list(filter(lambda x: (x[0]>=0) and (x[0]<width-0.5) and \
                                            (x[1]>=0) and (x[1]<height-0.5), process_pixel)))
    if len(temp_process_pixel.shape)==1:
        return img_cam

    process_pixel = temp_process_pixel[:,:2]
    process_cls_idx = temp_process_pixel[:,2]

    # round digits
    process_pixel = np.around(process_pixel)
    num_points, _ = np.shape(process_pixel)

    img_line = img_cam.copy()
    if line_width is None:
        l_thickness = cnf.line_thickness
    else:
        l_thickness = line_width
    for i in range(num_points):
        x_pix, y_pix = process_pixel[i,:].astype(int)

        if not (arr_cls_idx is None):
            color = cnf.cls_lane_color[int(process_cls_idx[i])]
        else:
            color = (255, 180, 0)

        img_line = cv2.line(img_line, (x_pix, y_pix), (x_pix, y_pix), color, l_thickness)

    return img_line

def get_camera_img_as_connected_line_with_projected_pixel(img_cam, pixels, arr_cls_idx=None, is_single_color=False):
    height, width, _ = np.shape(img_cam)
    process_pixel = pixels.copy()
    if arr_cls_idx is not None:
        num_points, _ = np.shape(pixels)
        process_cls_idx = np.reshape(arr_cls_idx.copy(),(num_points,1))
        process_pixel = np.concatenate((process_pixel, process_cls_idx), axis=1)
    temp_process_pixel = np.array(list(filter(lambda x: (x[0]>=0) and (x[0]<width-0.5) and \
                                            (x[1]>=0) and (x[1]<height-0.5), process_pixel)))
    process_pixel = temp_process_pixel[:,:2]
    process_cls_idx = temp_process_pixel[:,2]

    # round digits
    process_pixel = np.around(process_pixel)
    img_line = img_cam.copy()

    for j in range(6):
        idx_tuple = np.where(process_cls_idx == j)
        idx_tuple = idx_tuple[0]

        points_per_lane = process_pixel[idx_tuple,:]

        for i in range(len(points_per_lane)-1):
            x_curr, y_curr = points_per_lane[i,:].astype(int)
            x_next, y_next = points_per_lane[i+1,:].astype(int)
            if is_single_color:
                color = (0, 180, 255)
            else:
                color = cnf.cls_lane_color[j]
            
            img_line = cv2.line(img_line, (x_curr, y_curr), (x_next, y_next), color, cnf.line_thickness)

    return img_line

def get_corresponding_sample_from_dataset_type(runner, data_info, dataset_type):
    if dataset_type == 'AVELaneProjection':
        meta = data_info.copy()
        sample = dict()
        sample['meta'] = meta

        with open(meta['bev_tensor_label'], 'rb') as f:
            bev_tensor_label = pickle.load(f, encoding='latin1')
        sample['label'] = bev_tensor_label[:,0:144]

        pc = get_pc_os64_with_path(meta['pc'])
        pc = filter_pc_os64_with_roi(pc, runner.cfg.list_filter_roi, runner.cfg.filter_mode)
        pc = append_image_index_to_pc_os64(pc, runner.cfg.list_roi_xy, runner.cfg.list_grid_xy)

        sample['proj'] = get_projection_image_from_pointclouds(pc, is_flip=False)
        sample['proj'] = np.transpose(sample['proj'], (2,0,1))
        sample['proj'] = sample['proj'].astype(np.float32)

        # Make data to cuda
        sample['proj'] = sample['proj'][np.newaxis,:]
        sample['proj'] = torch.from_numpy(sample['proj']).cuda()

    elif dataset_type == 'AVELane':
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

        # Make input data to cuda
        sample['pillars'] = sample['pillars'][np.newaxis,:]
        sample['pillar_indices'] = sample['pillar_indices'][np.newaxis,:]
        sample['pillars'] = torch.from_numpy(sample['pillars']).cuda()
        sample['pillar_indices'] = torch.from_numpy(sample['pillar_indices']).cuda()
    elif dataset_type == 'KLane':
        meta = data_info.copy()
        sample = dict()
        sample['meta'] = meta

        with open(meta['bev_tensor_label'], 'rb') as f:
            bev_tensor_label = pickle.load(f, encoding='latin1')
        sample['label'] = bev_tensor_label[:,0:144]

        pc = get_pc_os64_with_path(meta['pc'])
        pc = filter_pc_os64_with_roi(pc, runner.cfg.list_filter_roi, runner.cfg.filter_mode)
        pc = append_image_index_to_pc_os64(pc, runner.cfg.list_roi_xy, runner.cfg.list_grid_xy)

        sample['proj'] = get_projection_image_from_pointclouds(pc, is_flip=False)
        sample['proj'] = np.transpose(sample['proj'], (2,0,1))
        sample['proj'] = sample['proj'].astype(np.float32)

        # Make data to cuda
        sample['proj'] = sample['proj'][np.newaxis,:]
        sample['proj'] = torch.from_numpy(sample['proj']).cuda()

    # Make label data to cuda
    sample['label'] = sample['label'][np.newaxis,:]
    sample['label'] = torch.from_numpy(sample['label']).cuda()

    return sample

def get_output_from_runner(runner, data_info, dataset_type, is_with_sample=False, is_get_features=False, is_measure_ms=False, is_get_attention_score=False):
    sample = get_corresponding_sample_from_dataset_type(runner, data_info, dataset_type)

    if is_get_features:
        output = runner.process_one_sample(sample, is_calc_f1=True, is_get_features=True, is_measure_ms=is_measure_ms)
    elif is_get_attention_score:
        output = runner.process_one_sample(sample, is_calc_f1=True, is_get_features=True, is_get_attention_score=is_get_attention_score)
    else:
        output = runner.process_one_sample(sample, is_calc_f1=True, is_measure_ms=is_measure_ms)

    if is_with_sample:
        return output, sample
    else:
        return output

def get_metrics_conf_cls(list_metric):
    metric_conf = np.array(list_metric[:2])
    metric_cls = np.array(list_metric[2:])
    
    return np.max(metric_conf), np.max(metric_cls)

def get_o3d_pointcloud(lane_map, pc_type):
    pcd = o3d.geometry.PointCloud()

    if pc_type in ['label', 'conf']:
        pc_xyz = get_point_cloud_from_bev_tensor_label(lane_map, with_cls=False, z_fix=getattr(cnf, 'z_fix_'+pc_type))
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        pcd.paint_uniform_color(getattr(cnf, 'pc_rgb_'+pc_type))
    elif pc_type == 'cls':
        pc_xyz, pc_cls = get_point_cloud_from_bev_tensor_label(lane_map, with_cls=True, z_fix=getattr(cnf, 'z_fix_'+pc_type))
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        list_cls_idx = []
        for idx in pc_cls.tolist():
            list_cls_idx.append(cnf.pc_rgb_cls[idx])
        pcd.colors = o3d.utility.Vector3dVector(np.array(list_cls_idx))
    
    return pcd

def get_wide_bev_map_with_white_background_for_vis(bev_map, thick_offset = 1, size_magnifier = 1.5):
    '''
    * in : bev map (0, 1, 2, 3, 4, 5, 255) 
    * out: bev map with magnified line for vis
    '''
    shape_label = np.shape(bev_map)

    bev_tensor_label_img = np.full(shape_label, 255, dtype=np.uint8)
    bev_tensor_label_img = cv2.cvtColor(bev_tensor_label_img, cv2.COLOR_GRAY2RGB)
    
    new_bev_tensor_label = np.full(shape_label, 255, dtype=np.uint8)
    for i in range(6):
        list_y, list_x = np.where(bev_map==i)
        for j in range(len(list_y)):
            for k in range(-thick_offset,thick_offset+1):
                new_x = list_x[j]+k
                if new_x < 0 or new_x > 143:
                    continue
                new_bev_tensor_label[list_y[j], new_x] = i
    # sys.exit()

    for i in range(6):
        bev_tensor_label_img[np.where(new_bev_tensor_label==i)] = cnf.cls_lane_color[i]

    bev_tensor_label_img = cv2.resize(bev_tensor_label_img, (0,0), fx=size_magnifier, fy=size_magnifier, interpolation=cv2.INTER_LINEAR)

    return bev_tensor_label_img

def get_white_background_label(p_frame, thick_offset = 1, size_magnifier = 1.5):
    if not p_frame.data_info:
        return None
    
    with open(p_frame.data_info['bev_tensor_label'], 'rb') as f:
        bev_tensor_label = pickle.load(f, encoding='latin1')
    bev_tensor_label = bev_tensor_label[:,0:144]
    shape_label = np.shape(bev_tensor_label)

    bev_tensor_label_img = np.full(shape_label, 255, dtype=np.uint8)
    bev_tensor_label_img = cv2.cvtColor(bev_tensor_label_img, cv2.COLOR_GRAY2RGB)
    
    new_bev_tensor_label = np.full(shape_label, 255, dtype=np.uint8)
    for i in range(6):
        list_y, list_x = np.where(bev_tensor_label==i)
        for j in range(len(list_y)):
            for k in range(-thick_offset,thick_offset+1):
                new_x = list_x[j]+k
                if new_x < 0 or new_x > 143:
                    continue
                new_bev_tensor_label[list_y[j], new_x] = i

    for i in range(6):
        bev_tensor_label_img[np.where(new_bev_tensor_label==i)] = cnf.cls_lane_color[i]

    bev_tensor_label_img = cv2.resize(bev_tensor_label_img, (0,0), fx=size_magnifier, fy=size_magnifier, interpolation=cv2.INTER_LINEAR)

    return bev_tensor_label_img

def get_projected_img_from_pre_calib(p_frame):
    if p_frame.data_info == None:
        return

    calib = p_frame.data_info['calib'].copy()
    intrinsic = calib[:4]
    extrinsic = calib[4:]

    rot, tra = get_rotation_and_translation_from_extrinsic(extrinsic)
    with open(p_frame.data_info['bev_tensor_label'], 'rb') as f:
        bev_tensor_label = pickle.load(f, encoding='latin1')
    pc_label, arr_cls_idx = get_point_cloud_from_bev_tensor_label(bev_tensor_label, with_cls=True)
    pc_label = get_pointcloud_with_rotation_and_translation(pc_label, rot, tra)
    pixel_label = get_pixel_from_point_cloud_in_camera_coordinate(pc_label, intrinsic)
    
    if p_frame.checkBox_inline.isChecked():
        img_with_line = get_camera_img_as_connected_line_with_projected_pixel(p_frame.cv_img, pixel_label, arr_cls_idx)
    else:
        img_with_line = get_camera_img_as_dot_with_projected_pixel(p_frame.cv_img, pixel_label, arr_cls_idx, 20)
    
    return img_with_line


