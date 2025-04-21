'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import numpy as np

__all__ = ['get_pc_os64_with_path', 'filter_pc_os64_with_roi', \
    'append_image_index_to_pc_os64', 'get_projection_image_from_pointclouds']

def get_pc_os64_with_path(path_pcd):
    f = open(path_pcd, 'r')
    lines = f.readlines()

    header = lines[:11]
    list_fields = header[2].split(' ')[1:]
    list_fields[-1] = list_fields[-1][:-1]
    points_with_fields = lines[11:]
    
    points_with_fields = list(map(lambda line: list(map(lambda x: float(x), line.split(' '))), points_with_fields))
    points_with_fields = np.array(points_with_fields)
    f.close()

    pc = dict()
    pc['path'] = path_pcd
    pc['values'] = points_with_fields
    pc['fields'] = list_fields

    return pc

def filter_pc_os64_with_roi(pc_os64, list_roi, filter_mode='xy'):
    if filter_mode == 'xy':
        return filter_pc_os64_with_roi_in_xy(pc_os64, list_roi)
    elif filter_mode == 'xyz':
        return filter_pc_os64_with_roi_in_xyz(pc_os64, list_roi)

def filter_pc_os64_with_roi_in_xy(pc_os64, list_roi_xy):
    x_min, x_max, y_min, y_max = list_roi_xy
    list_pc_values = pc_os64['values'].tolist()
    pc_os64['values'] = np.array(list(filter(lambda point: \
        (point[0] > x_min) and (point[0] < x_max) and \
        (point[1] > y_min) and (point[1] < y_max), list_pc_values)))
    
    return pc_os64

def filter_pc_os64_with_roi_in_xyz(pc_os64, list_roi_xyz):
    x_min, x_max, y_min, y_max, z_min, z_max = list_roi_xyz
    list_pc_values = pc_os64['values'].tolist()
    pc_os64['values'] = np.array(list(filter(lambda point: \
        (point[0] > x_min) and (point[0] < x_max) and \
        (point[1] > y_min) and (point[1] < y_max) and \
        (point[2] > z_min) and (point[2] < z_max), list_pc_values)))
    
    return pc_os64

def append_image_index_to_pc_os64(pc_os64, list_roi_xy, list_grid_xy):
    x_min, _, y_min, _ = list_roi_xy
    x_grid, y_grid = list_grid_xy

    list_xy_values = pc_os64['values'][:,:2].tolist()
    list_xy_values = list(map(lambda xy: [int((xy[0]-x_min)/x_grid), \
                                            int((xy[1]-y_min)/y_grid)], list_xy_values))
    
    arr_xy_values = np.array(list_xy_values)
    tuple_xy = (arr_xy_values[:,0], arr_xy_values[:,1])

    pc_os64.update({'img_idx': arr_xy_values})
    pc_os64.update({'img_idx_np_where': tuple_xy})

    return pc_os64

def get_projection_image_from_pointclouds(pc_os64, list_img_size_xy=[1152, 1152], list_value_idx = [2, 3, 4], \
                                            list_list_range = [[-2.0,1.5], [0,128], [0,32768]], is_flip=False):
    n_channels = len(list_value_idx)
    temp_img = np.full((list_img_size_xy[0], list_img_size_xy[1], n_channels), 0, dtype=float)

    list_list_values = []
    for channel_idx, value_idx in enumerate(list_value_idx):
        temp_arr = pc_os64['values'][:,value_idx].copy()

        v_min, v_max = list_list_range[channel_idx]
        temp_arr[np.where(temp_arr<v_min)] = v_min
        temp_arr[np.where(temp_arr>v_max)] = v_max
        temp_arr = (temp_arr-v_min)/(v_max-v_min)
        list_list_values.append(temp_arr)

        for idx, xy in enumerate(pc_os64['img_idx']):
            temp_img[xy[0], xy[1], channel_idx] = temp_arr[idx]

    if is_flip:
        temp_img = np.flip(np.flip(temp_img, 0), 1).copy()

    return temp_img
