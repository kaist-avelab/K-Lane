#!usr/bin/env python
# -*- coding: utf-8 -*-

# PyQt5
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import uic
from PyQt5.QtGui import *

# rospy
import rospy
from sensor_msgs.msg import Image
from lidar_msgs.msg import PillarTensorTraining
from lidar_msgs.msg import Params

# python2.7
import os
import inspect
import numpy as np
import pickle

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import cv2
from utils import *
from cv_bridge import CvBridge, CvBridgeError

form_path = "%s/mainframe.ui" % (os.path.dirname(os.path.abspath(__file__)))
form_class = uic.loadUiType(form_path)[0]

# Label status
_NONE = 0
_INITIALIZED = 1
_START = 2
_ING = 3
_END = 4

# Img pos
_NOT_IMG = 0
_TOP = 1
_LEFT = 2
_RIGHT = 3
_BOTTOM = 4
_MID = 5

# Debug
_DEBUG = True

# Sequence
_SEQUENCE = 1

class MainFrame(QMainWindow, form_class):
    def __init__(self):
        # python 2.7
        super(QMainWindow, self).__init__()
        self.setupUi(self)

        # Global Variable
        self._indicater_idx = None
        self._line_color_list = None

        self._is_labeling = False
        self._is_initialized = False
        self._label_status = _NONE
        self._line_num = 6 # 6: None, 0~5: idx
        self._x_b4 = None
        self._y_b4 = None

        # Image
        self._mat_line = None
        self._mat_bev_w_line = None

        # Undo
        self._mat_line_b4 = None
        self._mat_bev_w_line_b4 = None

        # Redo
        self._mat_line_bu = None
        self._mat_bev_w_line_bu = None

        # BEV Image
        self._bev_width = 576
        self._bev_height = 1152
        self._bev_x_grid = 4 # [pixel] 576/4 = 144
        self._bev_y_grid = 8 # [pixel] 1152/8 = 144

        # Time
        self._time_in_string = None

        # Label
        self._mat_bev_img_label = None
        self._mat_bev_tensor_label_rgb = None
        self._mat_bev_tensor_label = None

        self.setting_signal_and_ui()

    def setting_signal_and_ui(self):
        self.pushButtonLine_0.clicked.connect(self.func_push_button_line_0)
        self.pushButtonLine_1.clicked.connect(self.func_push_button_line_1)
        self.pushButtonLine_2.clicked.connect(self.func_push_button_line_2)
        self.pushButtonLine_3.clicked.connect(self.func_push_button_line_3)
        self.pushButtonLine_4.clicked.connect(self.func_push_button_line_4)
        self.pushButtonLine_5.clicked.connect(self.func_push_button_line_5)
        self.pushButtonEnd.clicked.connect(self.func_push_button_end)

        self.pushButtonInit.clicked.connect(self.func_push_button_init)
        self.pushButtonUndo.clicked.connect(self.func_push_button_undo)
        self.pushButtonRedo.clicked.connect(self.func_push_button_redo)
        self.pushButtonFinish.clicked.connect(self.func_push_button_finish)
        self.pushButtonSave.clicked.connect(self.func_push_button_save)
        self.pushButtonExit.clicked.connect(self.func_push_button_exit)

        self._line_color_list = [ ( 36,  36, 255), \
                                  (  0,  94, 255),\
                                  ( 54, 241, 255),\
                                  ( 22, 219,  29), \
                                  (255, 120,  36),\
                                  (255, 138, 201) ]

        back_ground = ');background:rgb(50,50,50)'

        self.pushButtonLine_0.setStyleSheet(''.join(['color:rgb(',str(self._line_color_list[0][2]), \
                                                              ',',str(self._line_color_list[0][1]), \
                                                              ',',str(self._line_color_list[0][0]),back_ground]))
        self.pushButtonLine_1.setStyleSheet(''.join(['color:rgb(',str(self._line_color_list[1][2]), \
                                                              ',',str(self._line_color_list[1][1]), \
                                                              ',',str(self._line_color_list[1][0]),back_ground]))
        self.pushButtonLine_2.setStyleSheet(''.join(['color:rgb(',str(self._line_color_list[2][2]), \
                                                              ',',str(self._line_color_list[2][1]), \
                                                              ',',str(self._line_color_list[2][0]),back_ground]))
        self.pushButtonLine_3.setStyleSheet(''.join(['color:rgb(',str(self._line_color_list[3][2]), \
                                                              ',',str(self._line_color_list[3][1]), \
                                                              ',',str(self._line_color_list[3][0]),back_ground]))
        self.pushButtonLine_4.setStyleSheet(''.join(['color:rgb(',str(self._line_color_list[4][2]), \
                                                              ',',str(self._line_color_list[4][1]), \
                                                              ',',str(self._line_color_list[4][0]),back_ground]))
        self.pushButtonLine_5.setStyleSheet(''.join(['color:rgb(',str(self._line_color_list[5][2]), \
                                                              ',',str(self._line_color_list[5][1]), \
                                                              ',',str(self._line_color_list[5][0]),back_ground]))
        self.pushButtonEnd.setStyleSheet(''.join(['color:rgb(255,255,255',back_ground]))

    def mousePressEvent(self, e):
        x = e.x()
        y = e.y()
        point_status = self.get_img_point_status(x,y)

        if _DEBUG:
            print(''.join(['x=',str(x),', y=',str(y),', ',self.get_img_point_status_as_str(point_status)]))

        if point_status == _TOP or point_status == _LEFT or point_status == _RIGHT \
            or point_status == _BOTTOM or point_status == _MID:
            # point in img
            if self._label_status == _START:
                self._x_b4, self._y_b4 = self.set_point_in_img(x,y,point_status)
                if _DEBUG:
                    print(''.join(['x_new=',str(self._x_b4),', y_new=',str(self._y_b4)]))
                self.set_logs(self._x_b4,self._y_b4,self._label_status,point_status)
                self._label_status = _ING
            elif self._label_status == _ING:
                x_new, y_new = self.set_point_in_img(x,y,point_status)
                if _DEBUG:
                    print(''.join(['x_new=',str(x_new),', y_new=',str(y_new)]))
                self.make_line_label(x_new, y_new, self._x_b4, self._y_b4, self._line_color)
                self._x_b4, self._y_b4 = x_new, y_new
                self.show_mat_bev_w_line()
                self.show_mat_line()
        else:
            if _DEBUG:
                print('not in img')

    def func_push_button_line_0(self):
        self.line_button_clicked(0)

    def func_push_button_line_1(self):
        self.line_button_clicked(1)

    def func_push_button_line_2(self):
        self.line_button_clicked(2)

    def func_push_button_line_3(self):
        self.line_button_clicked(3)

    def func_push_button_line_4(self):
        self.line_button_clicked(4)

    def func_push_button_line_5(self):
        self.line_button_clicked(5)

    def func_push_button_end(self):
        self._label_status = _END
        self.set_logs(self._x_b4,self._y_b4,self._label_status,self.get_img_point_status(self._x_b4, self._y_b4))

    def func_push_button_init(self):
        self._mat_line = np.zeros((self._bev_height, self._bev_width, 3), np.uint8)

        # if not mat_bev == None:
        self._mat_bev_w_line = mat_bev_shared.copy()
        self.show_mat_bev_w_line()
        self.show_mat_line()
        
        self._is_initialized = True
        self._label_status = _INITIALIZED
        self.set_logs(0,0,self._label_status,0)
        
    def func_push_button_undo(self):
        self.get_back_up_mat()
        self.show_mat_bev_w_line()
        self.show_mat_line()
        print('undo')

    def func_push_button_redo(self):
        self.get_redo_mat()
        self.show_mat_bev_w_line()
        self.show_mat_line()
        print('redo')

    def func_push_button_finish(self):
        self._mat_bev_img_label = np.zeros((self._bev_height, self._bev_width, 3), np.uint8)
        
        # Masking
        mat_line_num = np.zeros((self._bev_height, self._bev_width, 6), np.float)

        # No line filtering
        # Masking
        for k in range(6):
            if not self.get_is_line_not_exist(k):
                for j in range(self._bev_height):
                    for i in range(self._bev_width):
                        m_b, m_g, m_r = self._mat_line[j,i,:]
                        l_b, l_g, l_r = self._line_color_list[k]
                        if m_r == l_r and m_g == l_g and m_b == l_b:
                            self._mat_bev_img_label[j,i,:] = [l_b, l_g, l_r]
                            mat_line_num[j,i,k] = 1.
        self.labelBEVLabel.setPixmap(q_image_to_q_pixmap(cv_img_to_q_image(self._mat_bev_img_label)))

        # 144 x 144
        self._mat_bev_tensor_label = np.full((144,150), 255, dtype=np.uint8)
        x_grid = self._bev_x_grid # 4
        y_grid = self._bev_y_grid # 8

        for j in range(144): # row-wise
            for k in range(6): # line-wise
                temp = np.zeros(144)
                for i in range(144): # col-wise
                    # grid
                    for jj in range(y_grid):
                        for ii in range(x_grid):
                            temp[i] += mat_line_num[j*y_grid+jj,i*x_grid+ii,k]
                
                if np.sum(temp) == 0.:
                    self._mat_bev_tensor_label[j][144+k] = k
                else:
                    max_idx = np.argmax(temp)
                    self._mat_bev_tensor_label[j][max_idx] = k

        self._mat_bev_tensor_label_rgb = np.zeros((144,150,3), np.uint8)

        for j in range(144):
            for i in range(150):
                # print(self._mat_bev_img_label[j][i])
                idx = int(self._mat_bev_tensor_label[j][i])

                if idx == 255:
                    self._mat_bev_tensor_label_rgb[j,i,:] = [0,0,0]
                else:
                    self._mat_bev_tensor_label_rgb[j,i,:] = self._line_color_list[idx]

        mat_label_visualization = cv2.resize(self._mat_bev_tensor_label_rgb, dsize=(432,450),interpolation=cv2.INTER_NEAREST)
        self.labelBEVLabelFinished.setPixmap(q_image_to_q_pixmap(cv_img_to_q_image(mat_label_visualization)))

    def func_push_button_save(self):
        time_in_string = self._time_in_string

        file_path = '%s/temp' % (os.path.dirname(os.path.abspath(__file__)))
        file_seq = ''.join(['seq_',str(_SEQUENCE)])

        # tensor
        file_name_tensor_label = ''.join(['bev_tensor_label/bev_tensor_label_',time_in_string,'.pickle'])
        file_total_tensor_label = ''.join([file_path,'/',file_seq,'/',file_name_tensor_label])

        # bev_image
        file_name_img_label = ''.join(['bev_image_label/bev_image_label_',time_in_string,'.pickle'])
        file_total_img_label = ''.join([file_path,'/',file_seq,'/',file_name_img_label])

        if os.path.exists(file_total_tensor_label):
            main_frame.textEditLogs.append(''.join([file_name_tensor_label,' exists']))
        else:
            with open(file_total_tensor_label, 'wb') as f:
                pickle.dump(self._mat_bev_tensor_label, f, pickle.HIGHEST_PROTOCOL)
            main_frame.textEditLogs.append(''.join([file_name_tensor_label,' is saved']))

        if os.path.exists(file_total_img_label):
            main_frame.textEditLogs.append(''.join([file_name_img_label,' exists']))
        else:
            with open(file_total_img_label, 'wb') as f:
                pickle.dump(self._mat_bev_img_label, f, pickle.HIGHEST_PROTOCOL)
            main_frame.textEditLogs.append(''.join([file_name_img_label,' is saved']))

    def func_push_button_exit(self):
        self.close()

    def line_button_clicked(self, line_num):
        if not self._is_initialized:
            self.textEditLogs.append('Initailization is required')
            return

        if not self._label_status == _INITIALIZED and not self._label_status == _END: 
            self.textEditLogs.append('Ending the line is reqired')
            return

        self.set_back_up_mat()

        self._line_num = line_num
        self.textEditLogs.append(''.join(['Line ',str(line_num), \
                                            ' selected, Press the start point']))
        self._line_color = self._line_color_list[line_num]
        self._label_status = _START

    def get_img_point_status(self, x_pos, y_pos):
        pixel_offset = 8 # 8 pixel 정도 오프셋 when click top, bottom, left, right
        
        if x_pos >= 0 and x_pos < self._bev_width and \
           y_pos >= 0 and y_pos < pixel_offset :
            return _TOP
        elif x_pos >= 0 and x_pos < pixel_offset and \
             y_pos >= 0 and y_pos < self._bev_height-pixel_offset:
            return _LEFT
        elif x_pos >= self._bev_width-pixel_offset and x_pos < self._bev_width and \
             y_pos >= 0 and y_pos < self._bev_height-pixel_offset:
            return _RIGHT
        elif x_pos >= 0 and x_pos < self._bev_width and \
             y_pos >= self._bev_height-pixel_offset and y_pos < self._bev_height:
            return _BOTTOM
        elif x_pos >= pixel_offset and x_pos < self._bev_width-pixel_offset and \
             y_pos >= pixel_offset and y_pos < self._bev_height-pixel_offset:
            return _MID
        else:
            return _NOT_IMG # Re-click

    def get_img_point_status_as_str(self, point_status):
        if point_status == _TOP:
            return 'top'
        elif point_status == _LEFT:
            return 'left'
        elif point_status == _RIGHT:
            return 'right'
        elif point_status == _BOTTOM:
            return 'bottom'
        elif point_status == _MID:
            return 'mid'
        else:
            return 'img x'

    def set_point_top_edge(self, x, y):
        y_new = (y/self._bev_y_grid)*self._bev_y_grid
        return x, y_new

    def set_point_left_edge(self, x, y):
        x_new = (x/self._bev_x_grid)*self._bev_x_grid
        return x_new, y

    def set_point_right_edge(self, x, y):
        x_new = (x/self._bev_x_grid+1)*self._bev_x_grid-1
        return x_new, y

    def set_point_bottom_edge(self, x, y):
        y_new = (y/self._bev_y_grid+1)*self._bev_y_grid-1
        return x, y_new

    def set_point_in_img(self, x, y, point_status):
        if point_status == _TOP:
            return self.set_point_top_edge(x, y)
        elif point_status == _LEFT:
            return self.set_point_left_edge(x, y)
        elif point_status == _RIGHT:
            return self.set_point_right_edge(x, y)
        elif point_status == _BOTTOM:
            return self.set_point_bottom_edge(x, y)
        elif point_status == _MID:
            return x, y

    def make_line_label(self, x, y, x_b4, y_b4, line_color):
        cv2.line(self._mat_bev_w_line, (x,y), (x_b4, y_b4), line_color, thickness=1)
        cv2.line(self._mat_line, (x,y), (x_b4, y_b4), line_color, thickness=1)

    def show_mat_bev_w_line(self):
        self.labelBEVImage.setPixmap(q_image_to_q_pixmap(cv_img_to_q_image(self._mat_bev_w_line)))

    def show_mat_line(self):
        self.labelBEVLabel.setPixmap(q_image_to_q_pixmap(cv_img_to_q_image(self._mat_line)))

    def set_back_up_mat(self):
        self._mat_line_b4 = self._mat_line.copy()
        self._mat_bev_w_line_b4 = self._mat_bev_w_line.copy()

    def get_back_up_mat(self):
        self._mat_line_bu = self._mat_line.copy()
        self._mat_bev_w_line_bu = self._mat_bev_w_line.copy()
        self._mat_line = self._mat_line_b4.copy()
        self._mat_bev_w_line = self._mat_bev_w_line_b4.copy()

    def get_redo_mat(self):
        self._mat_line = self._mat_line_bu.copy()
        self._mat_bev_w_line = self._mat_bev_w_line_bu.copy()

    def get_is_line_not_exist(self, line_num):
        if line_num == 0:
            return self.checkBoxNoLine_0.isChecked()
        elif line_num == 1:
            return self.checkBoxNoLine_1.isChecked()
        elif line_num == 2:
            return self.checkBoxNoLine_2.isChecked()
        elif line_num == 3:
            return self.checkBoxNoLine_3.isChecked()
        elif line_num == 4:
            return self.checkBoxNoLine_4.isChecked()
        elif line_num == 5:
            return self.checkBoxNoLine_5.isChecked()

    def set_logs(self, x, y, label_status, point_status):
        if label_status == _INITIALIZED:
            self.textEditLogs.append(''.join(['Select line']))
        elif label_status == _START:
            self.textEditLogs.append(''.join(['Line ',str(self._line_num),\
                ' labeling start, start on ',self.get_img_point_status_as_str(point_status)]))
        elif label_status == _END:
            self.textEditLogs.append(''.join(['Line ',str(self._line_num),\
                ' labeling end, end on ',self.get_img_point_status_as_str(point_status)]))

def callback_pillar_tensor_training(msg_training):
    global mat_bev_shared

    bridge = CvBridge()

    try:
        mat_bev = bridge.imgmsg_to_cv2(msg_training.img_bev)
        mat_bev_shared = mat_bev.copy()
        mat_frontal = bridge.imgmsg_to_cv2(msg_training.img_frontal)
    except CvBridgeError as e:
        print(e)
    
    main_frame._time_in_string = msg_training.time_in_string
    main_frame.textEditLogs.append(''.join(['t = ', get_time_string_as_second(main_frame._time_in_string),', idx = ', str(msg_training.n_file_idx)]))

    tensors, bev_intensity_img = get_tensors_from_msg(msg_training)
    if main_frame.checkBoxSave.isChecked():
        save_tensors_as_pickle(tensors, bev_intensity_img, main_frame._time_in_string)

    main_frame.labelBEVImage.setPixmap(q_image_to_q_pixmap(cv_img_to_q_image(mat_bev)))
    main_frame.labelFrontalImage.setPixmap(q_image_to_q_pixmap(cv_img_to_q_image(cv_img_resize(mat_frontal,480,300))))

def get_tensors_from_msg(msg):
    data_f = msg.data_f
    layout_f = msg.layout_f
    layout_f_0 = layout_f.dim[0].size
    layout_f_1 = layout_f.dim[1].size
    layout_f_2 = layout_f.dim[2].size
    pillar_tensor = np.array(data_f).reshape(layout_f_0, layout_f_1, layout_f_2).astype(np.float32)

    data_n = msg.data_n
    layout_n = msg.layout_n
    layout_n_0 = layout_n.dim[0].size
    layout_n_1 = layout_n.dim[1].size
    pillar_idx_tensor = np.array(data_n).reshape(layout_n_0, layout_n_1).astype(np.int32)
    pillar_tensor = pillar_tensor[tf.newaxis, ...]
    pillar_idx_tensor = pillar_idx_tensor[tf.newaxis, ...]
    tensors = [pillar_tensor, pillar_idx_tensor]

    bev_f = msg.bev_f
    layout_bev_f = msg.layout_bev_f
    layout_bev_f_0 = layout_bev_f.dim[0].size
    layout_bev_f_1 = layout_bev_f.dim[1].size
    bev_intensity_img = np.array(bev_f).reshape(layout_bev_f_0, layout_bev_f_1).astype(np.float32)

    return tensors, bev_intensity_img

def save_tensors_as_pickle(tensors, bev_intensity_img, time_in_string):
    file_path = '%s/temp' % (os.path.dirname(os.path.abspath(__file__)))
    file_seq = ''.join(['seq_',str(_SEQUENCE)])

    # tensor
    file_name_tensor = ''.join(['bev_tensor/bev_tensor_',time_in_string,'.pickle'])
    file_total_tensor = ''.join([file_path,'/',file_seq,'/',file_name_tensor])

    # bev_image
    file_name_img = ''.join(['bev_image/bev_image_',time_in_string,'.pickle'])
    file_total_img = ''.join([file_path,'/',file_seq,'/',file_name_img])

    if os.path.exists(file_total_tensor):
        main_frame.textEditLogs.append(''.join([file_name_tensor,' exists']))
    else:
        with open(file_total_tensor, 'wb') as f:
            pickle.dump(tensors, f, pickle.HIGHEST_PROTOCOL)
        main_frame.textEditLogs.append(''.join([file_name_tensor,' is saved']))

    if os.path.exists(file_total_img):
        main_frame.textEditLogs.append(''.join([file_name_img,' exists']))
    else:
        with open(file_total_img, 'wb') as f:
            pickle.dump(bev_intensity_img, f, pickle.HIGHEST_PROTOCOL)
        main_frame.textEditLogs.append(''.join([file_name_img,' is saved']))

def get_time_string_as_second(time_in_string):
    time_sec = time_in_string[0:6]
    time_sec = time_sec.lstrip('0')
    time_nsec = time_in_string[7:]
    time_nsec = time_nsec.rstrip('0')
    time_string_sec = ''.join([time_sec,'.',time_nsec])
    
    return time_string_sec

app = QApplication(sys.argv) 
main_frame = MainFrame()

if __name__ == "__main__" :
    # Qt
    main_frame.show()

    # ROS
    rospy.init_node('gui_node', anonymous=True)
    
    rospy.Subscriber("pillar_tensor_training", PillarTensorTraining, callback_pillar_tensor_training)

    app.exec_()
