# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Tuple

import numpy as np
import torch
import trimesh

from mmdet3d.structures import (BaseInstance3DBoxes, Box3DMode,
                                CameraInstance3DBoxes, Coord3DMode,
                                DepthInstance3DBoxes, LiDARInstance3DBoxes)

def voxel2points(voxel, voxelSize=(0.4, 0.4, 0.4), range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], ignore_labels=[17, 255]):
    if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
    mask = torch.zeros_like(voxel, dtype=torch.bool)
    for ignore_label in ignore_labels:
        mask = torch.logical_or(voxel == ignore_label, mask)
    mask = torch.logical_not(mask)
    occIdx = torch.where(mask)
    points = torch.cat((occIdx[0][:, None] * voxelSize[0] + voxelSize[0] / 2 + range[0], \
                        occIdx[1][:, None] * voxelSize[1] + voxelSize[1] / 2 + range[1], \
                        occIdx[2][:, None] * voxelSize[2] + voxelSize[2] / 2 + range[2]), dim=1)
    return points, voxel[occIdx]

def voxel_profile(voxel, voxel_size=(0.4, 0.4, 0.4)):
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)
    # centers = voxel
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                          torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                          torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)

def generate_the_ego_car():
    ego_range = [-2, -1, 0, 2, 1, 1.5]
    ego_voxel_size=[0.1, 0.1, 0.1]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    ego_voxel_num = ego_xdim * ego_ydim * ego_zdim
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
    ego_points_label =  (np.ones((ego_point_xyz.shape[0]))*16).astype(np.uint8)
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    return ego_point_xyz

def my_compute_box_3d(center, size, heading_angle):
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    heading_angle = -heading_angle - math.pi / 2
    center[:, 2] = center[:, 2] + h / 2
    #R = rotz(1 * heading_angle)
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    #corners_3d = R @ torch.vstack([x_corners, y_corners, z_corners])
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d


def write_obj(points: np.ndarray, out_filename: str) -> None:
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()


def write_oriented_bbox(scene_bbox: np.ndarray, out_filename: str) -> None:
    """Export oriented (around Z axis) scene bbox to meshes.

    Args:
        scene_bbox (np.ndarray): xyz pos of center and 3 lengths
            (x_size, y_size, z_size) and heading angle around Z axis.
            Y forward, X right, Z upward, heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename (str): Filename.
    """

    def heading2rotmat(heading_angle: float) -> np.ndarray:
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(
            box: np.ndarray) -> trimesh.base.Trimesh:
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    if len(scene_bbox) == 0:
        scene_bbox = np.zeros((1, 7))
    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to obj file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='obj')


def to_depth_mode(
        points: np.ndarray,
        bboxes: BaseInstance3DBoxes) -> Tuple[np.ndarray, BaseInstance3DBoxes]:
    """Convert points and bboxes to Depth Coord and Depth Box mode."""
    if points is not None:
        points = Coord3DMode.convert_point(points.copy(), Coord3DMode.LIDAR,
                                           Coord3DMode.DEPTH)
    if bboxes is not None:
        bboxes = Box3DMode.convert(bboxes.clone(), Box3DMode.LIDAR,
                                   Box3DMode.DEPTH)
    return points, bboxes


# TODO: refactor lidar2img to img_meta
def proj_lidar_bbox3d_to_img(bboxes_3d: LiDARInstance3DBoxes,
                             input_meta: dict) -> np.ndarray:
    """Project the 3D bbox on 2D plane.

    Args:
        bboxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bbox in lidar coordinate
            system to visualize.
        input_meta (dict): Meta information.
    """
    corners_3d = bboxes_3d.corners.cpu().numpy()
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img = copy.deepcopy(input_meta['lidar2img']).reshape(4, 4)
    if isinstance(lidar2img, torch.Tensor):
        lidar2img = lidar2img.cpu().numpy()
    pts_2d = pts_4d @ lidar2img.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return imgfov_pts_2d


# TODO: remove third parameter in all functions here in favour of img_metas
def proj_depth_bbox3d_to_img(bboxes_3d: DepthInstance3DBoxes,
                             input_meta: dict) -> np.ndarray:
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes_3d (:obj:`DepthInstance3DBoxes`): 3D bbox in depth coordinate
            system to visualize.
        input_meta (dict): Meta information.
    """
    from mmdet3d.models import apply_3d_transformation
    from mmdet3d.structures import points_cam2img

    input_meta = copy.deepcopy(input_meta)
    corners_3d = bboxes_3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)

    # first reverse the data transformations
    xyz_depth = apply_3d_transformation(
        points_3d, 'DEPTH', input_meta, reverse=True)

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(xyz_depth,
                               xyz_depth.new_tensor(input_meta['depth2img']))
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return imgfov_pts_2d


# project the camera bboxes 3d to image
def proj_camera_bbox3d_to_img(bboxes_3d: CameraInstance3DBoxes,
                              input_meta: dict) -> np.ndarray:
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes_3d (:obj:`CameraInstance3DBoxes`): 3D bbox in camera coordinate
            system to visualize.
        input_meta (dict): Meta information.
    """
    from mmdet3d.structures import points_cam2img

    cam2img = copy.deepcopy(input_meta['cam2img'])
    corners_3d = bboxes_3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    assert (cam2img.shape == torch.Size([3, 3])
            or cam2img.shape == torch.Size([4, 4]))
    cam2img = cam2img.float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam2img)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return imgfov_pts_2d
