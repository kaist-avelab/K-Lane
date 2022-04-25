'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import os
import sys
abs_file_path = os.path.dirname(os.path.abspath(__file__))
# path where mainframe.ui exists
abs_file_path = abs_file_path.split('/')[:-1]
abs_file_path.extend(['baseline', 'vis'])
BASE_DIR = '/' + os.path.join(*abs_file_path)

window_size = [1290, 600] # width, height

# BGR Format to OpenCV
cls_lane_color = [
    (0, 0, 255),
    (0, 50, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 100)
]

# params to visulize on image
line_thickness = 15

# params of lidar coordinate
x_grid = 0.32
y_grid = 0.16
z_fix = -1.43 # grounds

# params to visualize in pointcloud
z_fix_label = -1.5
z_fix_conf = -1.1
z_fix_cls = -1.1

pc_rgb_label = [1, 0.706, 0]
pc_rgb_conf = [0, 0, 0] #[0.133, 0.545, 0.133] #[0.5, 0.99, 0] #[0, 0.75, 1]
pc_rgb_cls = []

for lane_color in cls_lane_color:
    temp = list(lane_color).copy()
    temp = list(map(lambda x: float(x)/255., reversed(temp)))
    pc_rgb_cls.append(temp)

### Setting Dataset Path Here ###
data_root = '/media/donghee/HDD_0/KLane'
### Setting Dataset Path Here ###

if __name__ == '__main__':
    print(abs_file_path)
    print(BASE_DIR)
