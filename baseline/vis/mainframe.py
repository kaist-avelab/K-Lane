'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import enum
from re import sub
import shutil
from cv2 import data
from numpy.core.fromnumeric import size
from baseline.utils import config
from numpy.lib.function_base import insert
import configs.config_vis as cnf

import torch
import open3d as o3d
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import uic
from PyQt5.QtGui import QIcon, QPixmap

from baseline.utils.vis_utils import *

form_path = './baseline/vis/mainframe.ui'
form_class = uic.loadUiType(form_path)[0]

class MainFrame(QMainWindow, form_class):
    def __init__(self, gpus):
        super().__init__()
        self.setupUi(self)
        self.gpus = gpus

        # Global values
        self.dataset = None
        self.data_info = None
        self.runner = None
        self.path_prev = None
        self.model_name = None

        self.initUI()

        self.show()

    def initUI(self):
        # Window
        self.setWindowTitle('LiDAR Lane Detection')
        self.resize(cnf.window_size[0], cnf.window_size[1])
        move_window_to_center(self)
        self.radioButton_test.setChecked(True)
        self.pushButton_load_ckpt.setDisabled(True)
        self.pushButton_inference.setDisabled(True)

        # Icon
        qpix_logo = QPixmap(os.path.join(cnf.BASE_DIR, 'logo.png'))
        self.label_logo.setPixmap(qpix_logo)

        # Signals
        self.pushButton_load.clicked.connect(self.push_button_load_dataset)
        self.listWidget_file.itemDoubleClicked.connect(self.list_widget_double_clicked)
        self.pushButton_load_config.clicked.connect(self.push_button_load_config)
        self.pushButton_load_ckpt.clicked.connect(self.push_button_load_ckpt)
        self.pushButton_calibrate.clicked.connect(self.push_button_calibrate)
        self.pushButton_save_calibration.clicked.connect(self.push_button_save_calibration)
        self.pushButton_inference.clicked.connect(self.push_button_inference)
        self.pushButton_pointcloud.clicked.connect(self.push_button_pointcloud)
        self.pushButton_add_desc.clicked.connect(self.push_button_add_description)
        self.pushButton_load_desc.clicked.connect(self.push_button_load_description)
        self.pushButton_save_desc.clicked.connect(self.push_button_save_description)
        self.pushButton_test_1.clicked.connect(self.push_button_test_1)
        self.pushButton_test_2.clicked.connect(self.push_button_test_2)
        self.pushButton_test_3.clicked.connect(self.push_button_test_3)
        self.pushButton_test_4.clicked.connect(self.push_button_test_4)
        self.pushButton_test_5.clicked.connect(self.push_button_test_5)
        self.pushButton_test_6.clicked.connect(self.push_button_test_6)
        self.pushButton_test_7.clicked.connect(self.push_button_test_7)
        self.pushButton_test_8.clicked.connect(self.push_button_test_8)
        self.pushButton_light_curve.clicked.connect(self.push_button_light_curve)
        self.pushButton_calibration_pre.clicked.connect(self.push_button_calibration_pre)
    
    def push_button_load_dataset(self):
        if self.radioButton_training.isChecked():
            split = 'train'
        elif self.radioButton_test.isChecked():
            split = 'test'
        self.dataset = load_dataset_vis(cnf.data_root, split, self.listWidget_file)

    def list_widget_double_clicked(self):
        current_item = self.listWidget_file.currentItem()
        self.textBrowser_logs.append(current_item.data(0) + ' is loaded')
        self.data_info = current_item.data(1)

        print(self.data_info.keys())
        # print(self.data_info[])
        self.cv_img = cv2.imread(self.data_info['frontal_img'])
        temp_front_img = get_q_pixmap_from_cv_img(self.cv_img, 768, 480)
        temp_front_img_small = get_q_pixmap_from_cv_img(self.cv_img, 480, 300)
        temp_label_img = get_rgb_img_from_path_label(self.data_info['bev_tensor_label'])
        temp_label_img = get_q_pixmap_from_cv_img(temp_label_img, 216, 216, cv2.INTER_LINEAR)
        
        for i in range(2):
            getattr(self, 'label_front_img_'+str(i+1)).setPixmap(temp_front_img)
            getattr(self, 'label_lane_label_'+str(i+1)).setPixmap(temp_label_img)
        getattr(self, 'label_front_img_'+str(3)).setPixmap(temp_front_img_small)
        getattr(self, 'label_lane_label_'+str(3)).setPixmap(temp_label_img)
        self.label_timestamp.setText(self.data_info['timestamp'])

    def push_button_load_config(self):
        if self.path_prev is None:
            path_config_folder = './configs'
        else:
            path_config_folder = os.path.join(*(self.path_prev.split('/')[:-1]))
        path_config = QFileDialog.getOpenFileName(self, 'Open config file', path_config_folder)[0]
        self.path_prev = path_config
        if not path_config:
            return

        self.model_name = path_config.split('/')[-1].split('.')[0]
        self.cfg, self.runner = load_config_and_runner(path_config, self.gpus)
        self.textBrowser_logs.append(f'* Config: [{path_config}] is loaded')
        self.pushButton_load_ckpt.setEnabled(True)

    def push_button_load_ckpt(self):
        if self.path_prev is None:
            path_ckpt_folder = './logs'
        else:
            path_ckpt_folder = os.path.join(*(self.path_prev.split('/')[:-1]))
        path_ckpt = QFileDialog.getOpenFileName(self, 'Open ckpt file', path_ckpt_folder)[0]
        self.path_prev = path_ckpt
        if not path_ckpt:
            return
        self.runner.load_ckpt(path_ckpt)
        self.textBrowser_logs.append(f'* ckpt: [{path_ckpt}] is loaded')
        self.pushButton_inference.setEnabled(True)

    def push_button_calibrate(self):
        is_single_color=True
        if self.data_info == None:
            return

        intrinsic, extrinsic = get_intrinsic_and_extrinsic_params_from_text_edit(\
                                                            get_list_p_text_edit(self))
        # print(intrinsic), print(extrinsic)
        rot, tra = get_rotation_and_translation_from_extrinsic(extrinsic)
        with open(self.data_info['bev_tensor_label'], 'rb') as f:
            bev_tensor_label = pickle.load(f, encoding='latin1')
        pc_label, arr_cls_idx = get_point_cloud_from_bev_tensor_label(bev_tensor_label, with_cls=True)
        pc_label = get_pointcloud_with_rotation_and_translation(pc_label, rot, tra)
        pixel_label = get_pixel_from_point_cloud_in_camera_coordinate(pc_label, intrinsic)
        
        if self.checkBox_inline.isChecked():
            if is_single_color:
                img_with_line = get_camera_img_as_connected_line_with_projected_pixel(self.cv_img, pixel_label, arr_cls_idx)#, is_single_color=True)
            else:
                img_with_line = get_camera_img_as_connected_line_with_projected_pixel(self.cv_img, pixel_label, arr_cls_idx)
        else:
            img_with_line = get_camera_img_as_dot_with_projected_pixel(self.cv_img, pixel_label, arr_cls_idx)
        temp_front_img = get_q_pixmap_from_cv_img(img_with_line, 768, 480)
        
        getattr(self, 'label_front_img_'+str(2)).setPixmap(temp_front_img)
        
    def push_button_save_calibration(self):
        intrinsic, extrinsic = get_intrinsic_and_extrinsic_params_from_text_edit(\
                                                            get_list_p_text_edit(self))
        path_calibration = QFileDialog.getSaveFileName(self, 'Save calibration file', './data/calib_seq_.txt', '')[0]
        if not path_calibration:
            return

        f = open(path_calibration, 'w')
        for temp in intrinsic:
            f.write(str(temp) + ',')
        for temp in extrinsic:
            f.write(str(temp) + ',')
        f.close()

        self.textBrowser_logs.append(f'* calibration: [{path_calibration}] is saved')

        ### checking ###
        # f = open(path_calibration, 'r')
        # val = f.readline()
        # val = list(map(lambda x: float(x), val.split(',')[:-1]))
        # intrinsic_out = val[:4]
        # extrinsic_out = val[4:]
        # print(intrinsic, extrinsic)
        # print(intrinsic_out, extrinsic_out)
        ### checking ###
    
    def push_button_inference(self):
        is_with_label = True
        is_save_img = True

        if (not self.runner) or (not self.data_info):
            return

        # print(self.runner.cfg.dataset_type)
        dataset_type = self.runner.cfg.dataset_type
        
        self.infer = get_output_from_runner(self.runner, self.data_info, dataset_type, is_measure_ms=True)
        # print(self.infer.keys())
        
        list_metric = ['accuracy', 'precision', 'recall', 'f1']
        for key_metric in list_metric:
            metric_conf, metric_cls = get_metrics_conf_cls(self.infer[key_metric])
            self.textBrowser_metric.append(f'* {key_metric}: conf={metric_conf}, cls={metric_cls}')
        list_f1 = self.infer['f1']
        
        print(f'* f1(conf): {list_f1[0]}, {list_f1[1]} / f1(cls): {list_f1[2]}, {list_f1[3]}')
        
        # str_description = '* datum description: '
        # for desc in self.data_info['description']:
        #     str_description += (desc+', ')
        # str_description = str_description[:-2]
        # self.textBrowser_logs.append(str_description)

        img_infer = self.cv_img.copy()

        val_calib = self.data_info['calib']
        intrinsic, extrinsic = val_calib[:4], val_calib[4:]
        rot, tra = get_rotation_and_translation_from_extrinsic(extrinsic)

        pc_label, arr_cls_idx = get_point_cloud_from_bev_tensor_label(self.infer['conf_cls_idx'], with_cls=True)
        pc_label = get_pointcloud_with_rotation_and_translation(pc_label, rot, tra)
        pixel_label = get_pixel_from_point_cloud_in_camera_coordinate(pc_label, intrinsic)
        img_infer = get_camera_img_as_dot_with_projected_pixel(img_infer, pixel_label, arr_cls_idx, line_width=20)

        if self.checkBox_vis_inference.isChecked():
            def get_thick_line(img_rgb, fx=None, fy=None):
                img_rgb_new = np.full_like(img_rgb, 255, dtype=np.uint8)
                idx_lane_ext = np.where((np.mean(img_rgb, axis=2))!=0)

                for idx_h, idx_w in zip(*idx_lane_ext):
                    img_rgb_new[idx_h,idx_w,:] = img_rgb[idx_h,idx_w,:]
                    img_rgb_new[idx_h,idx_w-1,:] = img_rgb[idx_h,idx_w,:] if idx_w-1 >= 0 else None
                    img_rgb_new[idx_h,idx_w+1,:] = img_rgb[idx_h,idx_w,:] if idx_w+1 < 144 else None

                return cv2.resize(img_rgb_new, (0,0), fx=fx, fy=fy) if not (fx is None) else img_rgb_new

            img_infer_img = cv2.resize(img_infer, (768,480), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("infer", img_infer_img)

            img_white_label = get_white_background_label(self, thick_offset=1, size_magnifier=1.0)
            img_white_label = cv2.resize(img_white_label, (0,0), fx=1.0, fy=2.0)
            cv2.imshow("label", img_white_label)

            # conf_pred = self.infer['conf_pred']
            # conf_pred = (conf_pred*255).astype(np.uint8)
            # conf_pred = cv2.cvtColor(conf_pred, cv2.COLOR_GRAY2BGR)
            # cv2.imshow("conf infer", conf_pred)

            rgb_bev_infer = self.infer['rgb_conf_cls_idx']
            rgb_bev_infer = get_thick_line(rgb_bev_infer, fx=1.0, fy=2.0)

            cv2.imshow("cls infer", rgb_bev_infer)

            if self.checkBox_vis_all.isChecked():
                seq = self.data_info['seq']
                tstamp = self.data_info['timestamp']
                img_header = f'{seq}_{tstamp}'
                cv2.imwrite(f'./imgs/{img_header}_front.png', img_infer_img)
                cv2.imwrite(f'./imgs/{img_header}_bev.png', rgb_bev_infer)
                cv2.imwrite(f'./imgs/{img_header}_label.png', img_white_label)

        if self.checkBox_pc_label.isChecked():
            img_white_label = get_white_background_label(self, thick_offset=1, size_magnifier=4.0)
            offset_black_pixel = 12
            h_label, w_label, _ = img_white_label.shape

            img_infer[:2*offset_black_pixel+h_label,:2*offset_black_pixel+w_label,:] = 0
            img_infer[offset_black_pixel:offset_black_pixel+h_label,offset_black_pixel:offset_black_pixel+w_label,:]= img_white_label

        # if is_save_img:
        #     path_img = './imgs'
        #     timestamp = self.data_info['timestamp']
        #     name_img = f'{path_img}/{self.model_name}_{timestamp}.png'
        #     img_save = cv2.resize(img_infer, (768,480), interpolation=cv2.INTER_LINEAR)
        #     cv2.imwrite(name_img, img_save)

        temp_front_img = get_q_pixmap_from_cv_img(img_infer, 768, 480)
        getattr(self, 'label_front_img_'+str(1)).setPixmap(temp_front_img)

    def push_button_pointcloud(self, is_label_with_color=True):
        current_point_cloud = o3d.io.read_point_cloud(self.data_info['pc'])

        list_pc = [current_point_cloud]

        is_label_with_color = False

        if self.checkBox_pc_label.isChecked() and not (self.data_info == None):
            with open(self.data_info['bev_tensor_label'], 'rb') as f:
                bev_tensor_label = pickle.load(f, encoding='latin1')
            if is_label_with_color:
                list_pc.append(get_o3d_pointcloud(bev_tensor_label, pc_type='cls'))
            else:
                list_pc.append(get_o3d_pointcloud(bev_tensor_label, pc_type='label'))

        if self.checkBox_pc_conf.isChecked() and not (self.infer == None):
            bev_infer_conf = self.infer['conf_pred'].copy()
            bev_infer_conf[np.where(bev_infer_conf==0.0)]=255
            bev_infer_conf[np.where(bev_infer_conf==1.0)]=0
            list_pc.append(get_o3d_pointcloud(bev_infer_conf, pc_type='conf'))

        if self.checkBox_pc_cls.isChecked() and not (self.infer == None):
            bev_infer_cls = self.infer['conf_cls_idx'].copy()
            list_pc.append(get_o3d_pointcloud(bev_infer_cls, pc_type='cls'))

        o3d.visualization.draw_geometries(list_pc,
			front = [ -0.84117093347047334, -0.028460638431088793, 0.54001986328699747 ],
			lookat = [ 23.251048651552424, 3.7226471525807079, 5.7020016107635776 ],
		    up = [ 0.54050619535691036, -0.013100711585632797, 0.84123803060533786 ],
			zoom = 0.059999999999999998)

    def push_button_add_description(self):
        if self.data_info is None:
            return
        desc = ''
        desc = desc + self.data_info['timestamp']
        desc = desc + ', ' + str(self.spinBox_num_lanes.value())

        list_desc = ['curve', 'merging', 'daylight', 'night', 'urban', 'highway']
        for temp in list_desc:
            if getattr(self, 'checkBox_des_'+ temp).isChecked():
                desc = desc + ', ' + temp
        
        for occ_num in range(7):
            if getattr(self, f'radioButton_des_occ_{occ_num}').isChecked():
                desc = desc + ', ' + f'occ{occ_num}'

        self.plainTextEdit_desc.appendPlainText(desc)

    def push_button_load_description(self):
        path_desc = QFileDialog.getOpenFileName(self, 'Open description file', './data/description_frames.txt')[0]
        if not path_desc:
            return

        f = open(path_desc, 'r')
        lines_all = f.read()
        self.plainTextEdit_desc.appendPlainText(lines_all)
        f.close()

    def push_button_save_description(self):
        path_desc = QFileDialog.getSaveFileName(self, 'Save description file', './data/description_frames.txt', '')[0]
        if not path_desc:
            return
        
        f = open(path_desc, 'w')
        plain_text = self.plainTextEdit_desc.toPlainText()
        f.write(plain_text)
        f.close()

        path_desc_lightcurve = os.path.join(*(path_desc.split('.')[:-1]))
        f = open(f'{path_desc_lightcurve}_lightcurve.txt', 'w')
        plain_text_light_curve = self.plainTextEdit_desc_light_curve.toPlainText()
        f.write(plain_text_light_curve)
        f.close()

    def push_button_light_curve(self):
        if self.data_info is None:
            return
        desc = ''
        desc = desc + self.data_info['timestamp']

        self.plainTextEdit_desc_light_curve.appendPlainText(desc)

    ### Customized Functions ###
    def push_button_test_1(self):
        if (not self.runner) or (not self.data_info):
            return

    def push_button_test_2(self):
        if (not self.runner) or (not self.data_info):
            return

    def push_button_test_3(self):
        if (not self.runner) or (not self.data_info):
            return

    def push_button_test_4(self):
        if (not self.runner) or (not self.data_info):
            return
    
    def push_button_test_5(self):
        if (not self.runner) or (not self.data_info):
            return

    def push_button_test_6(self):
        if (not self.runner) or (not self.data_info):
            return

    def push_button_calibration_pre(self):
        if (not self.runner) or (not self.data_info):
            return

    def push_button_test_7(self):
        if (not self.runner) or (not self.data_info):
            return

    def push_button_test_8(self):
        if (not self.runner) or (not self.data_info):
            return
