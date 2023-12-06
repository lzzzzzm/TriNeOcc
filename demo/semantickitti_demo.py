# Copyright (c) OpenMMLab. All rights reserved.
import nerfacc
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from mmdet3d.datasets.transforms.loading import LoadPointsFromFile, LoadAnnotations3D, PointSegClassMapping
from projects.TPVFormer.tpvformer import SegLabelMapping
from projects.TriNeOcc.trineocc import SemanticKittiOccDataset, LoadKittiImageFromFile, LoadRaysFromMultiViewImage, LoadDepthsFromPoints


from mmdet3d.datasets.transforms.formating import Pack3DDetInputs

from mmdet3d.visualization import Det3DLocalVisualizer
import mmcv

import torch
import torch.nn as nn

def _generate_semantickitti_dataset_config():
    data_root = 'data/semanticskitti'
    ann_file = 'semantickitti_infos_train.pkl'

    dataset_type = 'SemanticKittiOccDataset'
    data_root = 'data/semantickitti/'
    class_names = [
        'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist',
        'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
        'fence', 'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign'
    ]
    labels_map = {
        0: 19,  # "unlabeled"
        1: 19,  # "outlier" mapped to "unlabeled" --------------mapped
        10: 0,  # "car"
        11: 1,  # "bicycle"
        13: 4,  # "bus" mapped to "other-vehicle" --------------mapped
        15: 2,  # "motorcycle"
        16: 4,  # "on-rails" mapped to "other-vehicle" ---------mapped
        18: 3,  # "truck"
        20: 4,  # "other-vehicle"
        30: 5,  # "person"
        31: 6,  # "bicyclist"
        32: 7,  # "motorcyclist"
        40: 8,  # "road"
        44: 9,  # "parking"
        48: 10,  # "sidewalk"
        49: 11,  # "other-ground"
        50: 12,  # "building"
        51: 13,  # "fence"
        52: 19,  # "other-structure" mapped to "unlabeled" ------mapped
        60: 8,  # "lane-marking" to "road" ---------------------mapped
        70: 14,  # "vegetation"
        71: 15,  # "trunk"
        72: 16,  # "terrain"
        80: 17,  # "pole"
        81: 18,  # "traffic-sign"
        99: 19,  # "other-object" to "unlabeled" ----------------mapped
        252: 0,  # "moving-car" to "car" ------------------------mapped
        253: 6,  # "moving-bicyclist" to "bicyclist" ------------mapped
        254: 5,  # "moving-person" to "person" ------------------mapped
        255: 7,  # "moving-motorcyclist" to "motorcyclist" ------mapped
        256: 4,  # "moving-on-rails" mapped to "other-vehic------mapped
        257: 4,  # "moving-bus" mapped to "other-vehicle" -------mapped
        258: 3,  # "moving-truck" to "truck" --------------------mapped
        259: 4  # "moving-other"-vehicle to "other-vehicle"-----mapped
    }

    metainfo = dict(
        classes=class_names, seg_label_mapping=labels_map, max_label=259)

    backend_args=None
    pipeline = [
        dict(
            type=LoadPointsFromFile,
            coord_type='LIDAR',
            load_dim=4,
            use_dim=3,
            backend_args=backend_args
        ),
        dict(type=LoadKittiImageFromFile,
             data_root=data_root),
        dict(
            type=LoadAnnotations3D,
            with_bbox_3d=False,
            with_label_3d=False,
            with_seg_3d=True,
            with_occ_3d=False,
            seg_3d_dtype='np.int32',
            dataset_type='semantickitti',
            seg_offset=2 ** 16
        ),
        dict(type=PointSegClassMapping),
        dict(type=LoadDepthsFromPoints,
             depth_min=1.0,
             depth_max=60.0),
        dict(
            type=LoadRaysFromMultiViewImage,
            render=False,
            use_radii=False,
            to_world=False,
            select_rgb_rays_number=32,
            select_rays_number=32,
        ),
        dict(
            type=Pack3DDetInputs,
            keys=['img', 'points', 'rays_bundle', 'pts_semantic_mask'],
            meta_keys=['lidar2img', 'lidar2ego', 'ego2lidar', 'cam2img', 'cam2lidar', 'cam2ego']
        )
    ]
    modality = dict(use_lidar=True, use_camera=True)
    return data_root, ann_file, pipeline, modality, metainfo

def gradient_point_cloud_color_map(points):
    # 根据距离生成色彩
    colors = np.zeros([points.shape[0], 3])
    # 使用x,y计算到中心点的距离
    dist = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))

    dist_max = np.max(dist)
    print(f"dist_max: {dist_max}")
    # 调整渐变半径
    dist = dist / 60  # 我这里的半径是51.2m，
    # dist = dist / 2

    # RGB
    min = [127, 0, 255]  # 紫色
    max = [255, 255, 0]  # 黄色

    # 最近处为紫色
    # colors[:,0] = 127
    # colors[:,2] = 255

    # 减R(127 -> 0),加G(0->255),再减B(255->0)，再加R(0 -> 255)
    # 127+255+255+255
    all_color_value = 127 + 255 + 255 + 255
    dist_color = dist * all_color_value

    # 减R (127 -> 0)
    clr_1 = 127
    dy_r = 127 - dist_color
    tmp = np.zeros([colors[dist_color < clr_1].shape[0], 3])
    tmp[:, 0] = dy_r[dist_color < clr_1]
    tmp[:, 1] = 0
    tmp[:, 2] = 255
    colors[dist_color < clr_1] = tmp

    # 加G (0->255)
    clr_2 = 127 + 255
    dy_g = dist_color - clr_1
    tmp = np.zeros([colors[(dist_color >= clr_1) & (dist_color < clr_2)].shape[0], 3])
    tmp[:, 0] = 0
    tmp[:, 1] = dy_g[(dist_color >= clr_1) & (dist_color < clr_2)]
    tmp[:, 2] = 255
    colors[(dist_color >= clr_1) & (dist_color < clr_2)] = tmp

    # 减B (255->0)
    clr_3 = 127 + 255 + 255
    dy_b = dist_color - clr_2
    tmp = np.zeros([colors[(dist_color >= clr_2) & (dist_color < clr_3)].shape[0], 3])
    tmp[:, 0] = 0
    tmp[:, 1] = 255
    tmp[:, 2] = dy_b[(dist_color >= clr_2) & (dist_color < clr_3)]
    colors[(dist_color >= clr_2) & (dist_color < clr_3)] = tmp

    # 加R(0 -> 255)
    clr_4 = 127 + 255 + 255 + 255
    dy_r = dist_color - clr_3
    tmp = np.zeros([colors[(dist_color >= clr_3) & (dist_color < clr_4)].shape[0], 3])
    tmp[:, 0] = dy_r[(dist_color >= clr_3) & (dist_color < clr_4)]
    tmp[:, 1] = 255
    tmp[:, 2] = 0
    colors[(dist_color >= clr_3) & (dist_color < clr_4)] = tmp

    '''
    '''
    # 外围都为黄色
    tmp = np.zeros([colors[dist_color > clr_4].shape[0], 3])
    tmp[:, 0] = 255
    tmp[:, 1] = 255
    tmp[:, 2] = 0
    colors[dist_color > clr_4] = tmp

    points = np.concatenate((points[:, :3], colors), axis=1)

    return points

def tensor2array(data):
    return data.cpu().numpy()

def draw_sem_pts_3d(points, pts_sem, palette, ignore_index=0):
    vis = Det3DLocalVisualizer()
    vis._draw_pts_sem_seg(points, pts_sem, palette, ignore_index=0)
    vis.show()
    vis.o3d_vis.destroy_window()

def select_points_in_frustum(points_2d, x1, y1, x2, y2):
    """
    Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
    :param points_2d: point cloud projected into 2D
    :param points_3d: point cloud
    :param x1: left bound
    :param y1: upper bound
    :param x2: right bound
    :param y2: lower bound
    :return: points (2D and 3D) that are in the frustum
    """
    keep_ind = (points_2d[:, 0] > x1) * \
               (points_2d[:, 1] > y1) * \
               (points_2d[:, 0] < x2) * \
               (points_2d[:, 1] < y2)

    return keep_ind

def draw_pts_on_image(points, img, lidar2img, pts_sem=None, palette=None):
    depth_max = 60
    depth_min = 0
    height, width = img.shape[:2]
    points_img = np.matmul(points, lidar2img[:3, :3].T) + np.expand_dims(lidar2img[:3, 3], axis=0)
    points_img = np.concatenate(
        [points_img[:, :2] / (points_img[:, 2:3] + 1e-6), points_img[:, 2:3]],
        axis=1
    )
    # get depth map
    depth = points_img[:, 2]
    points_coor = np.floor(points_img)
    keep_index = (points_coor[:, 0] >= 0) & (points_coor[:, 0] < width) & \
                 (points_coor[:, 1] >= 0) & (points_coor[:, 1] < height) \
                 & (depth < depth_max) & (depth >= depth_min)
    keep_points = np.array(points_coor[keep_index], dtype=np.int32)

    valid_points_img = points_img[keep_index]
    valid_depths = depth[keep_index]

    img = img.astype(np.uint8)
    # vis depth
    plt.imshow(img)
    plt.scatter(valid_points_img[:, 0], valid_points_img[:, 1], c=valid_points_img[:, 2], cmap='rainbow_r', alpha=0.5, s=2)
    plt.show()
    # vis sem
    pts_semantic_mask = tensor2array(pts_sem.pts_semantic_mask)
    pts_semantic_mask = np.array(pts_semantic_mask, dtype=np.uint8)
    valid_pts_semantic_mask = pts_semantic_mask[keep_index]
    palette = np.array(palette, dtype=np.uint8)
    pts_color = palette[pts_semantic_mask]
    plt.figure()
    plt.imshow(img)
    plt.scatter(valid_points_img[:, 0], valid_points_img[:, 1], c=valid_pts_semantic_mask, cmap='rainbow_r', alpha=0.5,
                s=2)
    plt.show()

def draw_rays_on_3d(points, rays_bundle):
    vis = Det3DLocalVisualizer()

    origins, directions, viewdirs = rays_bundle[0][:, :3].cuda(), rays_bundle[0][:, 3:6].cuda(), rays_bundle[
                                                                                                     0][:,
                                                                                                 6:9].cuda()
    ray_indices, t_starts, t_ends = nerfacc.ray_marching(origins, viewdirs, near_plane=1.0, far_plane=40.0,
                                                         render_step_size=1.0, stratified=True)
    t_mid = (t_starts + t_ends) / 2.0
    sample_locs = origins[ray_indices] + t_mid * viewdirs[ray_indices]
    colors = gradient_point_cloud_color_map(sample_locs.cpu().numpy())
    vis.set_points(points, vis_mode='add', pcd_mode=2)
    vis.set_points(colors, vis_mode='add', pcd_mode=2, mode='xyzrgb')
    vis.show()
    vis.o3d_vis.destroy_window()

def main():
    np.random.seed(0)
    data_root, ann_file, pipeline, modality, metainfo = \
        _generate_semantickitti_dataset_config()

    sem_kitti_dataset = SemanticKittiOccDataset(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=pipeline)
    palette = sem_kitti_dataset.METAINFO['palette']
    np.random.seed()
    sample_number = np.random.randint(0, len(sem_kitti_dataset))
    data = sem_kitti_dataset.prepare_data(sample_number)
    # vis points on 3D
    points = tensor2array(data['inputs']['points'])
    pts_sem = data['data_samples'].gt_pts_seg
    draw_sem_pts_3d(points, pts_sem, palette)
    # vis points on img
    img = tensor2array(data['inputs']['img'].permute(0, 2, 3, 1))
    pts2img = data['data_samples'].metainfo['lidar2img']
    draw_pts_on_image(points, img[0], lidar2img=pts2img, pts_sem=pts_sem, palette=palette)
    # vis rays
    rays_bundle = data['inputs']['rays_bundle']
    draw_rays_on_3d(points, rays_bundle)



if __name__ == '__main__':
    main()


