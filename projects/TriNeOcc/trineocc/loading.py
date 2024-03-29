# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Union

import mmcv
import numpy as np
import torch
import torch.nn.functional as F

from mmcv.transforms.base import BaseTransform
from mmengine.fileio import get

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.structures import points_cam2img
from mmdet3d.registry import TRANSFORMS

Number = Union[int, float]


@TRANSFORMS.register_module()
class LoadDepthsFromPoints(BaseTransform):
    """Map original semantic class to valid category ids.

    Required Keys:

    - seg_label_mapping (np.ndarray)
    - pts_semantic_mask (np.ndarray)

    Added Keys:

    - points (np.float32)

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    """

    def __init__(self,
                 depth_min,
                 depth_max) -> None:
        self.depth_min = depth_min
        self.depth_max = depth_max

    def transform(self, results: dict) -> dict:
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
            Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'points' in results
        points_lidar = results['points'].tensor.numpy()
        points_lidar_classes = results['pts_semantic_mask']
        lidar2img = np.array(results['lidar2img'], dtype=np.float32)
        height, width = results['img'][0].shape[:2]

        semantics_maps, depth_maps = [], []
        for index, l2i in enumerate(lidar2img):
            # init map
            depth_map = np.zeros((height, width), dtype=np.float32)
            semantics_map = np.full(shape=(height, width), fill_value=255, dtype=np.float32)

            # get valid points_img
            points_i = np.matmul(points_lidar, l2i[:3, :3].T) + np.expand_dims(l2i[:3, 3], axis=0)
            points_i = np.concatenate(
                [points_i[:, :2] / (points_i[:, 2:3] + 1e-6), points_i[:, 2:3]],
                axis=1
            )
            # get depth map
            depth = points_i[:, 2]

            points_coor = np.floor(points_i)
            keep_index = (points_coor[:, 0] >= 0) & (points_coor[:, 0] < width) & \
                         (points_coor[:, 1] >= 0) & (points_coor[:, 1] < height) \
                         & (depth < self.depth_max) & (depth >= self.depth_min)

            keep_points = np.array(points_coor[keep_index], dtype=np.int32)
            keep_points_lidar_classes = points_lidar_classes[keep_index]
            keep_depth = depth[keep_index]

            depth_map[keep_points[:, 1], keep_points[:, 0]] = keep_depth
            semantics_map[keep_points[:, 1], keep_points[:, 0]] = keep_points_lidar_classes

            depth_maps.append(depth_map)
            semantics_maps.append(semantics_map)

        depth_maps = np.stack(depth_maps)
        semantics_maps = np.stack(semantics_maps)

        results['depth_maps'] = depth_maps
        results['semantics_maps'] = semantics_maps

        return results


@TRANSFORMS.register_module()
class BEVOccLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``TPVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'
        - 'ego2lidar‘

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename, cam2img, lidar2cam, cam2lidar, lidar2img, cam2ego = [], [], [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])
            cam2ego.append(cam_item['cam2ego'])
            lidar2cam_array = np.array(cam_item['lidar2cam']).astype(
                np.float32)
            lidar2cam_rot = lidar2cam_array[:3, :3]
            lidar2cam_trans = lidar2cam_array[:3, 3:4]
            camera2lidar = np.eye(4)
            camera2lidar[:3, :3] = lidar2cam_rot.T
            camera2lidar[:3, 3:4] = -1 * np.matmul(
                lidar2cam_rot.T, lidar2cam_trans.reshape(3, 1))
            cam2lidar.append(camera2lidar)

            cam2img_array = np.eye(4).astype(np.float32)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img']).astype(
                np.float32)
            cam2img.append(cam2img_array)
            lidar2img.append(cam2img_array @ lidar2cam_array)

        lidar2ego = np.array(results['lidar_points']['lidar2ego'])
        lidar2ego_rot = lidar2ego[:3, :3]
        lidar2ego_trans = lidar2ego[:3, 3:4]
        ego2lidar = np.eye(4)
        ego2lidar[:3, :3] = lidar2ego_rot.T
        ego2lidar[:3, 3:4] = -1 * np.matmul(
            lidar2ego_rot.T, lidar2ego_trans.reshape(3, 1)
        )

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['cam2ego'] = np.stack(cam2ego, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['cam2lidar'] = np.stack(cam2lidar, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)
        results['lidar_points']['ego2lidar'] = ego2lidar

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        # gbr follow tpvformer
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results


@TRANSFORMS.register_module()
class LoadRaysFromMultiViewImage(BaseTransform):

    def __init__(self,
                 render,
                 select_rgb_rays_number,
                 select_rays_number) -> None:
        self.render = render
        self.select_rays_number = select_rays_number
        self.select_rgb_rays_number = select_rgb_rays_number

    def get_rays_from_single_image(self, c2i, c2w, valid_index_x=None, valid_index_y=None, H=None, W=None,
                                   render=False):
        if not render:
            if H is not None and W is not None:
                x = torch.randint(0, W, size=(self.select_rgb_rays_number,))
                y = torch.randint(0, H, size=(self.select_rgb_rays_number,))
            else:
                sample_index = np.random.randint(0, valid_index_x.shape[0], size=(self.select_rays_number,))
                x = torch.tensor(valid_index_x[sample_index])
                y = torch.tensor(valid_index_y[sample_index])
        else:
            x, y = torch.meshgrid(
                torch.arange(W),
                torch.arange(H),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        c2w = torch.tensor(c2w)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - c2i[0, 2] + 0.5) / c2i[0, 0],
                    (y - c2i[1, 2] + 0.5) / c2i[1, 1],
                ],
                dim=-1,
            ),
            (0, 1),
            value=1.0,
        )  # [num_rays, 3]

        directions = (camera_dirs[:, None, :] * c2w[:3, :3]).sum(dim=-1).to(torch.float32)
        origins = torch.broadcast_to(c2w[:3, 3], directions.shape).to(torch.float32)
        # normalized directions
        viewdirs = (directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )).to(torch.float32)

        return origins, directions, viewdirs, x, y

    def render_transform(self, results: dict) -> Optional[dict]:
        img_height, img_width = results['img'][0].shape[0], results['img'][0].shape[1]
        rays_bundle = []
        for c2i, c2w in zip(results['cam2img'], results['cam2ego']):
            origins, directions, viewdirs, x, y = self.get_rays_from_single_image(c2i, c2w, H=img_height, W=img_width,
                                                                                  render=True)
            rays = torch.cat([origins, directions, viewdirs], -1)
            rays_bundle.append(rays)

        rays_bundle = torch.stack(rays_bundle)
        results['rays_bundle'] = rays_bundle
        return results

    def train_transform(self, results: dict) -> Optional[dict]:
        index = 0
        img_height, img_width = results['img'][0].shape[0], results['img'][0].shape[1]
        rays_bundle, depth_maps, semantic_maps, rgb_maps = [], [], [], []
        for c2i, c2w, depth_map, semantic_map in zip(results['cam2img'], results['cam2ego'], results['depth_maps'],
                                                     results['semantics_maps']):
            valid_index_y, valid_index_x = np.where(semantic_map != 255)
            origins, directions, viewdirs, x, y = self.get_rays_from_single_image(c2i, c2w, valid_index_x=valid_index_x,
                                                                                  valid_index_y=valid_index_y)
            semantic_map = semantic_map[y, x]
            depth_map = depth_map[y, x]
            if self.select_rgb_rays_number:
                img = results['img'][index]
                rgb_map = img[y, x] / 255.0
                rgb_maps.append(rgb_map)
            depth_maps.append(depth_map)
            semantic_maps.append(semantic_map)
            rays = torch.cat([origins, directions, viewdirs, x.unsqueeze(-1), y.unsqueeze(-1)], -1)
            rays_bundle.append(rays)

        rays_bundle = torch.stack(rays_bundle)
        depth_maps = np.stack(depth_maps)
        semantic_maps = np.stack(semantic_maps)
        if self.select_rgb_rays_number:
            rgb_maps = np.stack(rgb_maps)
            results['rgb_maps'] = rgb_maps

        results['rays_bundle'] = rays_bundle
        results['depth_maps'] = depth_maps
        results['semantics_maps'] = semantic_maps
        return results

    def transform(self, results: dict) -> Optional[dict]:
        if not self.render:
            results = self.train_transform(results)
        else:
            results = self.render_transform(results)
        return results
