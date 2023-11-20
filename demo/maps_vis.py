# Copyright (c) OpenMMLab. All rights reserved.
import nerfacc
import cv2 as cv
import numpy as np

from mmdet3d.datasets.transforms.loading import LoadPointsFromFile, LoadAnnotations3D
from projects.TPVFormer.tpvformer import SegLabelMapping
from projects.TriNeOcc.trineocc import NuScenesOccDataset, LoadRaysFromMultiViewImage, BEVOccLoadMultiViewImageFromFiles, LoadDepthsFromPoints, S3IMLoss


from mmdet3d.datasets.transforms.formating import Pack3DDetInputs

from mmdet3d.visualization import Det3DLocalVisualizer


from mmengine.visualization.utils import tensor2ndarray
import torch
import torch.nn as nn


palette = [
    [0, 0, 0],  # noise                black
    [255, 120, 50],  # barrier              orange
    [255, 192, 203],  # bicycle              pink
    [255, 255, 0],  # bus                  yellow
    [0, 150, 245],  # car                  blue
    [0, 255, 255],  # construction_vehicle cyan
    [255, 127, 0],  # motorcycle           dark orange
    [255, 0, 0],  # pedestrian           red
    [255, 240, 150],  # traffic_cone         light yellow
    [135, 60, 0],  # trailer              brown
    [160, 32, 240],  # truck                purple
    [255, 0, 255],  # driveable_surface    dark pink
    [139, 137, 137],  # other_flat           dark red
    [75, 0, 75],  # sidewalk             dard purple
    [150, 240, 80],  # terrain              light green
    [230, 230, 250],  # manmade              white
    [0, 175, 0],  # vegetation           green
]
palette = np.array(palette)

def _generate_nus_dataset_config():
    data_root = 'data/nuscenes'
    ann_file = 'nuscenes_infos_train.pkl'
    classes = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]

    point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    backend_args=None
    pipeline = [
        dict(
            type=LoadPointsFromFile,
            coord_type='LIDAR',
            load_dim=5,
            use_dim=3,
            backend_args=backend_args
        ),
        dict(
            type=BEVOccLoadMultiViewImageFromFiles
        ),
        dict(
            type=LoadAnnotations3D,
            with_bbox_3d=False,
            with_label_3d=False,
            with_seg_3d=True,
            with_occ_3d=True,
            seg_3d_dtype='np.uint8'
        ),
        dict(type=SegLabelMapping),
        dict(type=LoadDepthsFromPoints,
             depth_min=1.0,
             depth_max=60.0),
        dict(
            type=LoadRaysFromMultiViewImage,
            render=False,
            select_rgb_rays_number=512,
            select_rays_number=512,
        ),
        dict(
            type=Pack3DDetInputs,
            keys=['img', 'points', 'rays_bundle','pts_semantic_mask', 'depth_maps', 'semantics_maps', 'rgb_maps', 'occ_semantics', 'occ_mask_camera'],
            meta_keys=['lidar2img', 'lidar2ego', 'ego2lidar', 'cam2img', 'cam2lidar', 'cam2ego']
        )
    ]
    modality = dict(use_lidar=True, use_camera=True)
    data_prefix = dict(
        pts='samples/LIDAR_TOP',
        pts_semantic_mask='lidarseg/v1.0-mini',
        CAM_FRONT='samples/CAM_FRONT',
        CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
        CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
        CAM_BACK='samples/CAM_BACK',
        CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
        CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
        sweeps='sweeps/LIDAR_TOP')
    return data_root, ann_file, classes, data_prefix, pipeline, modality


def main():
    np.random.seed(0)
    data_root, ann_file, classes, data_prefix, pipeline, modality = \
        _generate_nus_dataset_config()

    nus_dataset = NuScenesOccDataset(
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=data_prefix,
        pipeline=pipeline)
    np.random.seed()
    sample_number = np.random.randint(0, len(nus_dataset))
    data = nus_dataset.prepare_data(sample_number)
    rays_bundle = data['inputs']['rays_bundle']
    x, y = rays_bundle[..., 9].to(torch.int64), rays_bundle[..., 10].to(torch.int64)
    semantics_maps = data['data_samples'].gt_maps['semantics_maps'].to(torch.int64)
    semantics_maps = tensor2ndarray(semantics_maps)
    H, W = data['inputs']['img'][0].shape[1], data['inputs']['img'][0].shape[2]
    loss_fn = S3IMLoss(patch_height=H, patch_width=W)
    for i in range(x.shape[0]):
        semantics_map = np.zeros(shape=(H, W, 3))
        x_index = x[i]
        y_index = y[i]
        semantics_color = palette[semantics_maps[i]]
        semantics_map[y_index, x_index] = semantics_color
        semantics_map = torch.tensor(semantics_map).reshape(H*W, 3)
        loss = loss_fn(semantics_map, semantics_map+1)
        print(loss)
        # cv.imshow('img',  semantics_map)
        # cv.waitKey()


    # vis._draw_occ_sem_seg(occ_semantics, nus_dataset.METAINFO['palette'])
    # vis.show()


if __name__ == '__main__':
    main()


