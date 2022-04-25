'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import open3d as o3d
import numpy as np

__all__ = ['get_pc_os64_with_path', 'filter_pc_os64_with_roi', \
    'append_image_index_to_pc_os64', 'get_projection_image_from_pointclouds']

def get_pc_os64_with_path(path_pcd):
    '''
    *  in: pcd file, e.g., /media/donghee/T5/MMLDD/train/seq_1/pc/pc_001270427447090.pcd
    * out: Pointcloud dictionary
    *       keys: 'path',   'points',   'fields'
    *       type: str,      np.array,   list 
    '''
    f = open(path_pcd, 'r')
    lines = f.readlines()

    header = lines[:11]
    list_fields = header[2].split(' ')[1:]
    list_fields[-1] = list_fields[-1][:-1]
    # num_points = int(header[-2].split(' ')[1])
    points_with_fields = lines[11:]

    # assert num_points == len(points_with_fields), \
    #     f'The number of points is not {num_points}'
    
    points_with_fields = list(map(lambda line: list(map(lambda x: float(x), line.split(' '))), points_with_fields))
    points_with_fields = np.array(points_with_fields)
    f.close()

    pc = dict()
    pc['path'] = path_pcd
    pc['values'] = points_with_fields
    pc['fields'] = list_fields

    return pc

def filter_pc_os64_with_roi(pc_os64, list_roi, filter_mode='xy'):
    '''
    *  in: Pointcloud dictionary
    *       e.g., list roi xy: [x min, x max, y min, y max], meter in LiDAR coords
    * out: Pointcloud dictionary
    '''
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
    '''
    *  in: Pointcloud dictionary
    *       list roi xy: [x min, x max, y min, y max], meter in LiDAR coords
    *       list grid xy: [x grid, y grid], meter in LiDAR coords
    * out: Pointcloud dictionary
    *       keys: 'path',   'points',   'fields',   'img_coords'
    *       type: str,      np.array,   list,       np.array
    '''
    x_min, _, y_min, _ = list_roi_xy
    x_grid, y_grid = list_grid_xy

    list_xy_values = pc_os64['values'][:,:2].tolist()
    list_xy_values = list(map(lambda xy: [int((xy[0]-x_min)/x_grid), \
                                            int((xy[1]-y_min)/y_grid)], list_xy_values))
    
    # np.where convention
    arr_xy_values = np.array(list_xy_values)
    tuple_xy = (arr_xy_values[:,0], arr_xy_values[:,1])

    pc_os64.update({'img_idx': arr_xy_values})
    pc_os64.update({'img_idx_np_where': tuple_xy})

    return pc_os64

def get_projection_image_from_pointclouds(pc_os64, list_img_size_xy=[1152, 1152], list_value_idx = [2, 3, 4], \
                                            list_list_range = [[-2.0,1.5], [0,128], [0,32768]], is_flip=False):
    '''
    *  in: Pointcloud dictionary with 'img_idx'
    * out: Image
            value: 0 ~ 1 normalized by list range
            type: float
    '''
    n_channels = len(list_value_idx)
    temp_img = np.full((list_img_size_xy[0], list_img_size_xy[1], n_channels), 0, dtype=float)

    list_list_values = [] # z, intensity, reflectivity
    for channel_idx, value_idx in enumerate(list_value_idx):
        temp_arr = pc_os64['values'][:,value_idx].copy()

        # Normalize
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

if __name__ == '__main__':
    import os
    list_pc = os.listdir('/media/donghee/T5/MMLDD/train/seq_4/pc/')
    pc = get_pc_os64_with_path('/media/donghee/T5/MMLDD/train/seq_4/pc/' + list_pc[210])
    print(pc['values'].shape)
    pc = filter_pc_os64_with_roi(pc, [0.02, 46.08, -11.52, 11.52, -2.0, 1.5], filter_mode='xyz')
    print(pc['values'].shape)
    pc = append_image_index_to_pc_os64(pc, [0, 46.08, -11.52, 11.52], [0.04, 0.02])

    print(pc['fields'])
    print(np.min(pc['values'], axis=0))
    print(np.max(pc['values'], axis=0))

    import cv2
    img_projection = get_projection_image_from_pointclouds(pc)
    cv2.imshow('img_c0', img_projection[:,:,0])
    print(np.max(img_projection[:,:,0]), np.min(img_projection[:,:,0]))
    cv2.imshow('img_c1', img_projection[:,:,1])
    print(np.max(img_projection[:,:,1]), np.min(img_projection[:,:,1]))
    cv2.imshow('img_c2', img_projection[:,:,2])
    print(np.max(img_projection[:,:,2]), np.min(img_projection[:,:,2]))
    cv2.imshow('img', img_projection)
    cv2.waitKey(0)

    # import matplotlib.pyplot as plt
    # # plt.hist(pc['values'][:,3], bins='auto')
    # plt.hist(pc['values'][:,4], bins='auto')
    # plt.show()    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc['values'][:,:3])
    o3d.visualization.draw_geometries([pcd])
