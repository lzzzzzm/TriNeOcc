# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet3d.datasets.transforms.loading import LoadPointsFromFile, LoadAnnotations3D
from projects.TPVFormer.tpvformer import SegLabelMapping
from projects.TriNeOcc.trineocc import NuScenesOccDataset, LoadRaysFromMultiViewImage, BEVOccLoadMultiViewImageFromFiles, LoadDepthsFromPoints


from mmdet3d.datasets.transforms.formating import Pack3DDetInputs

from mmdet3d.visualization import Det3DLocalVisualizer

import nerfacc

import torch

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
            select_rays_number=64,
        ),
        dict(
            type=Pack3DDetInputs,
            keys=['img', 'points', 'rays_bundle','pts_semantic_mask', 'depth_maps', 'semantics_maps', 'occ_semantics', 'occ_mask_camera'],
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
    points = data['inputs']['points']
    rays_bundle = data['inputs']['rays_bundle']
    occ_semantics = data['data_samples'].gt_occ_seg.occ_semantics

    vis = Det3DLocalVisualizer()
    device = 'cuda'
    # for index in range(6):
    #     scene_aabb = torch.tensor([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], device=device, dtype=torch.float32)
    #     origins, directions, viewdirs = rays_bundle[index][:, :3].cuda(), rays_bundle[index][:, 3:6].cuda(), rays_bundle[index][:, 6:].cuda()
    #     ray_indices, t_starts, t_ends = nerfacc.ray_marching(origins, viewdirs, scene_aabb=scene_aabb, render_step_size=0.2, stratified=True)
    #     t_mid = (t_starts + t_ends) / 2.0
    #     sample_locs = origins[ray_indices] + t_mid * viewdirs[ray_indices]
    #     vis.set_points(sample_locs.cpu().numpy(), vis_mode='add', pcd_mode=2)
    # vis bug
    # origins, directions, viewdirs = rays_bundle[5][:, :3].cuda(), rays_bundle[5][:, 3:6].cuda(), rays_bundle[5][:, 6:].cuda()
    # ray_indices, t_starts, t_ends = nerfacc.ray_marching(origins, viewdirs, scene_aabb=scene_aabb, render_step_size=0.2,
    #                                                      stratified=True)
    # t_mid = (t_starts + t_ends) / 2.0
    # sample_locs = origins[ray_indices] + t_mid * viewdirs[ray_indices]
    # vis.set_points(sample_locs.cpu().numpy(), vis_mode='add', pcd_mode=2)
    vis.set_points(points.cpu().numpy(), vis_mode='add')
    # vis._draw_occ_sem_seg(occ_semantics, nus_dataset.METAINFO['palette'])
    vis.show()


if __name__ == '__main__':
    main()


