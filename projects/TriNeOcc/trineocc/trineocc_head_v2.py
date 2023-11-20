import mmcv
import mmengine
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS

from torch_efficient_distloss import flatten_eff_distloss
import nerfacc

import numpy as np
import cv2 as cv

from collections import OrderedDict


def img2mse(x, y):
    return torch.mean((x - y) * (x - y))


camvid_colors = OrderedDict([
    (0, torch.tensor([0, 0, 0], dtype=torch.uint8, device='cuda')),  # noise     black
    (1, torch.tensor([128, 0, 0], dtype=torch.uint8, device='cuda')),  # barrier   dark red
    (2, torch.tensor([0, 128, 0], dtype=torch.uint8, device='cuda')),  # bicycle   dark green
    (3, torch.tensor([128, 128, 0], dtype=torch.uint8, device='cuda')),  # bus
    (4, torch.tensor([0, 0, 128], dtype=torch.uint8, device='cuda')),  # car       dark blue
    (5, torch.tensor([128, 0, 128], dtype=torch.uint8, device='cuda')),  # construction_vehicle
    (6, torch.tensor([64, 0, 192], dtype=torch.uint8, device='cuda')),  # motorcycle
    (7, torch.tensor([192, 128, 128], dtype=torch.uint8, device='cuda')),  # pedestrian light pink
    (8, torch.tensor([64, 0, 0], dtype=torch.uint8, device='cuda')),  # traffic_cone  brown
    (9, torch.tensor([64, 64, 128], dtype=torch.uint8, device='cuda')),  # trailer
    (10, torch.tensor([128, 0, 192], dtype=torch.uint8, device='cuda')),  # truck
    (11, torch.tensor([128, 64, 0], dtype=torch.uint8, device='cuda')),  # driveable_surface
    (12, torch.tensor([128, 128, 64], dtype=torch.uint8, device='cuda')),  # other_flat
    (13, torch.tensor([192, 0, 128], dtype=torch.uint8, device='cuda')),  # sidewalk      pink
    (14, torch.tensor([128, 64, 64], dtype=torch.uint8, device='cuda')),  # terrain
    (15, torch.tensor([64, 192, 128], dtype=torch.uint8, device='cuda')),  # manmade
    (16, torch.tensor([0, 192, 0], dtype=torch.uint8, device='cuda')),  # vegetation    green
    (17, torch.tensor([128, 64, 128], dtype=torch.uint8, device='cuda'))])  # free


@MODELS.register_module()
class TriNeOccHeadV2(BaseModule):

    def __init__(self,
                 tpv_h,
                 tpv_w,
                 tpv_z,
                 scene_aabb,
                 near_planes,
                 far_planes,
                 voxel_size=0.4,
                 render_step_size=0.2,
                 density_threshold=0.05,
                 num_classes=18,
                 in_dims=64,
                 position_dim=3,
                 hidden_dims=128,
                 out_dims=None,
                 scale_h=2,
                 scale_w=2,
                 scale_z=2,
                 supervised_rgb=True,
                 aug_infer=False,
                 ignore_index=0,
                 distloss_weight=0,
                 s3imloss_weight=1.0,
                 nerf_decoder=None,
                 # ffpe=None,
                 loss_s3im=None,
                 loss_semantics=None,
                 loss_depth=None):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.num_classes = num_classes
        # render params
        self.position_dim = position_dim
        self.distloss_weight = distloss_weight
        self.s3imloss_weight = s3imloss_weight
        self.density_threshold = density_threshold
        self.voxel_size = voxel_size
        self.scene_aabb = torch.tensor(scene_aabb, device='cuda')
        self.render_step_size = render_step_size
        self.supervised_rgb = supervised_rgb
        self.aug_infer = aug_infer

        self.min_bound = self.scene_aabb[:3]
        self.max_bound = self.scene_aabb[3:]
        self.near_planes = near_planes
        self.far_planes = far_planes

        out_dims = in_dims if out_dims is None else out_dims
        self.in_dims = in_dims

        self.nerf_decoder = MODELS.build(nerf_decoder)
        # self.ffpe = MODELS.build(ffpe)
        self.loss_s3im = MODELS.build(loss_s3im)
        self.loss_semantics = MODELS.build(loss_semantics)
        self.loss_depth = MODELS.build(loss_depth)
        self.ignore_index = ignore_index

    def forward(self, tpv_list, points=None):
        pass

    def query_point_feautres(self,
                             tpv_list,
                             tpv_scale,
                             voxel_size,
                             origins=None,
                             viewdirs=None,
                             ray_indices=None,
                             t_starts=None,
                             t_ends=None,
                             positions=None):
        tpv_hw, tpv_zh, tpv_wz = tpv_list

        if positions is None:
            t_origins = origins[ray_indices]  # (n_samples, 3)
            t_dirs = viewdirs[ray_indices]  # (n_samples, 3)
            distances = (t_starts + t_ends) / 2.0
            positions = t_origins + t_dirs * distances

        positions = torch.clamp(positions, self.min_bound, self.max_bound + 1e-6)
        point_coors = positions.clone()
        point_coors[..., 0] = (point_coors[..., 0] - self.min_bound[0]) / point_coors.new_tensor(voxel_size[0])
        point_coors[..., 1] = (point_coors[..., 1] - self.min_bound[1]) / point_coors.new_tensor(voxel_size[1])
        point_coors[..., 2] = (point_coors[..., 2] - self.min_bound[2]) / point_coors.new_tensor(voxel_size[2])
        # normalized to -1~1
        point_coors = point_coors.reshape(1, 1, -1, 3)
        point_coors[
            ...,
            0] = point_coors[..., 0] / (self.tpv_w * tpv_scale[0]) * 2 - 1
        point_coors[
            ...,
            1] = point_coors[..., 1] / (self.tpv_h * tpv_scale[1]) * 2 - 1
        point_coors[
            ...,
            2] = point_coors[..., 2] / (self.tpv_z * tpv_scale[2]) * 2 - 1

        sample_loc = point_coors[..., [0, 1]]
        tpv_hw_pts = F.grid_sample(
            tpv_hw, sample_loc, align_corners=False)
        sample_loc = point_coors[..., [1, 2]]
        tpv_zh_pts = F.grid_sample(
            tpv_zh, sample_loc, align_corners=False)
        sample_loc = point_coors[..., [2, 0]]
        tpv_wz_pts = F.grid_sample(
            tpv_wz, sample_loc, align_corners=False)

        tpv_featurs = (tpv_hw_pts + tpv_zh_pts + tpv_wz_pts).squeeze(0).squeeze(1).transpose(0, 1)
        return point_coors.squeeze(0).squeeze(0), tpv_featurs

    def render_img(self, tpv_list, batch_rays_bundle, batch_data_samples):
        tpv_hw, tpv_zh, tpv_wz = tpv_list
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        render_maps = {'depth_maps':[], 'rgb_maps':[], 'semantics_maps':[]}

        for i, data_sample in enumerate(batch_data_samples):
            rays_bundle = batch_rays_bundle[i]
            H, W = data_sample.batch_input_shape[0], data_sample.batch_input_shape[1]
            for cam_num in range(rays_bundle.shape[0]):
                origins, directions, viewdirs = rays_bundle[cam_num][:, :3], rays_bundle[cam_num][:, 3:6], rays_bundle[
                                                                                                               cam_num][
                                                                                                           :, 6:]
                total_number = origins.shape[0]
                render_depth_maps = []
                render_semantics_maps = []
                render_rgb_maps = []
                for chunk_number in mmengine.track_iter_progress(range(0, total_number, 1024)):
                    chunk_origins = origins[chunk_number:chunk_number + 1024]
                    chunk_viewdirs = viewdirs[chunk_number:chunk_number + 1024]
                    ray_indices, t_starts, t_ends = nerfacc.ray_marching(chunk_origins, chunk_viewdirs,
                                                                         near_plane=self.near_planes[cam_num],
                                                                         far_plane=self.far_planes[cam_num],
                                                                         render_step_size=self.render_step_size,
                                                                         stratified=False)
                    distances = (t_starts + t_ends) / 2
                    point_coors, tpv_featurs = self.query_point_feautres(
                        tpv_list=[tpv_hw, tpv_zh, tpv_wz],
                        tpv_scale=[1.0, 1.0, 1.0],
                        voxel_size=[0.4, 0.4, 0.4],
                        origins=chunk_origins,
                        viewdirs=chunk_viewdirs,
                        ray_indices=ray_indices,
                        t_starts=t_starts,
                        t_ends=t_ends
                    )
                    fused_featurs = self.ffpe(point_coors, tpv_featurs)
                    input_viewdirs = self.ffpe.view_freq_embed(chunk_viewdirs[ray_indices])
                    semantics, rgbs, density = self.nerf_decoder(fused_featurs, input_viewdirs=input_viewdirs)
                    weights = nerfacc.render_weight_from_density(
                        t_starts,
                        t_ends,
                        density,
                        ray_indices=ray_indices,
                        n_rays=origins[chunk_number:chunk_number + 1024].shape[0],
                    ).to(torch.float64)
                    render_depths = nerfacc.accumulate_along_rays(
                        weights,
                        ray_indices,
                        values=distances,
                        n_rays=origins[chunk_number:chunk_number + 1024].shape[0],
                    ).flatten()
                    render_semantics = nerfacc.accumulate_along_rays(
                        weights,
                        ray_indices,
                        values=semantics,
                        n_rays=origins[chunk_number:chunk_number + 1024].shape[0])
                    render_rgbs = nerfacc.accumulate_along_rays(
                        weights,
                        ray_indices,
                        values=rgbs,
                        n_rays=origins[chunk_number:chunk_number + 1024].shape[0])
                    pred_semantics = render_semantics.argmax(-1)
                    semantics_rgb = torch.zeros(size=(pred_semantics.shape[0], 3), device=pred_semantics.device,
                                                dtype=torch.uint8)
                    for label_class in range(self.num_classes - 1):
                        index = torch.where(pred_semantics == label_class)[0]
                        if index is None:
                            continue
                        semantics_rgb[index] = camvid_colors[label_class]

                    render_depth_maps.append(render_depths * 5)
                    render_semantics_maps.append(semantics_rgb)
                    render_rgb_maps.append(render_rgbs * 255)

                render_depth_maps = torch.stack(render_depth_maps).reshape(H, W).cpu().numpy()
                render_rgb_maps = torch.stack(render_rgb_maps).reshape(H, W, 3).cpu().numpy()
                render_rgb_maps = np.array(render_rgb_maps, dtype=np.uint8)
                render_semantics_maps = torch.stack(render_semantics_maps).reshape(H, W, 3).cpu().numpy()
                render_maps['depth_maps'].append(render_depth_maps)
                render_maps['rgb_maps'].append(render_rgb_maps)
                render_maps['semantics_maps'].append(render_semantics_maps)

        return render_maps

    def predict_seg(self, tpv_list, batch_points, batch_data_samples):
        tpv_hw, tpv_zh, tpv_wz = tpv_list
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)
        seg_preds = []
        for i, data_sample in enumerate(batch_data_samples):
            # change lidar-coord to ego-coord
            points = batch_points[i]
            lidar2ego = points.new_tensor(data_sample.lidar2ego)
            points = torch.cat((points, torch.ones_like(points[..., :1])), -1)
            ego_points = (points @ (lidar2ego.T))[:, :3]

            point_coors, tpv_featurs = self.query_point_feautres(
                tpv_list=[tpv_hw, tpv_zh, tpv_wz],
                tpv_scale=[1.0, 1.0, 1.0],
                voxel_size=[0.4, 0.4, 0.4],
                positions=ego_points
            )
            fused_featurs = self.ffpe(point_coors, tpv_featurs)
            semantics, pred_density = self.nerf_decoder(fused_featurs)
            seg_pred = semantics.argmax(-1)
            seg_preds.append(seg_pred)
        return seg_preds

    def predict_occ(self, tpv_list, batch_data_samples):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        tpv_hw, tpv_zh, tpv_wz = tpv_list
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        occ_preds = []
        for i, data_sample in enumerate(batch_data_samples):
            # get the ego coord predposition
            occ_pred = torch.full(size=(16 * 200 * 200,), fill_value=self.num_classes - 1,
                                  device='cuda')
            center_coords = torch.zeros(size=(16 * 200 * 200, 3)).cuda()
            center_coord_x = (torch.arange(self.scene_aabb[0], self.scene_aabb[3],
                                           self.voxel_size) + self.voxel_size / 2).unsqueeze(-1).expand(-1,
                                                                                                        200 * 16).reshape(
                -1)
            center_coord_y = (torch.arange(self.scene_aabb[1], self.scene_aabb[4],
                                           self.voxel_size) + self.voxel_size / 2).unsqueeze(-1).expand(-1,
                                                                                                        16).reshape(
                -1).unsqueeze(-1).T.expand(200, -1).reshape(-1)
            center_coord_z = (torch.arange(self.scene_aabb[2], self.scene_aabb[5] - 1e-6,
                                           self.voxel_size) + self.voxel_size / 2).unsqueeze(-1).expand(-1,
                                                                                                        200 * 200).T.reshape(
                -1)
            center_coords[:, 0] = center_coord_x
            center_coords[:, 1] = center_coord_y
            center_coords[:, 2] = center_coord_z
            if self.aug_infer:
                voting_semantics = torch.full(size=(7, 16 * 200 * 200), fill_value=self.num_classes - 1,
                                              device='cuda')
                voting_density = torch.zeros(size=(7, 16 * 200 * 200), device='cuda')
                (center_xl_coords, center_xr_coords,
                 center_yl_coords, center_yr_coords,
                 center_zl_coords, center_zr_coords) = (center_coords.clone(), center_coords.clone(),
                                                        center_coords.clone(), center_coords.clone(),
                                                        center_coords.clone(), center_coords.clone())

                center_xl_coords[:, 0] = center_xl_coords[:, 0] - self.voxel_size / 4
                center_xr_coords[:, 0] = center_xr_coords[:, 0] + self.voxel_size / 4
                center_yl_coords[:, 1] = center_yl_coords[:, 1] - self.voxel_size / 4
                center_yr_coords[:, 1] = center_yr_coords[:, 1] + self.voxel_size / 4
                center_zl_coords[:, 2] = center_zl_coords[:, 2] - self.voxel_size / 4
                center_zr_coords[:, 2] = center_zr_coords[:, 2] + self.voxel_size / 4

                coords = torch.stack(
                    [center_coords, center_xl_coords, center_xr_coords, center_yl_coords, center_yr_coords,
                     center_zl_coords, center_zr_coords])
                for index in range(coords.shape[0]):
                    point_coors, tpv_featurs = self.query_point_feautres(
                        tpv_list=[tpv_hw, tpv_zh, tpv_wz],
                        tpv_scale=[1.0, 1.0, 1.0],
                        voxel_size=[0.4, 0.4, 0.4],
                        positions=coords[index]
                    )
                    fused_featurs = self.ffpe(point_coors, tpv_featurs)
                    decoder_featurs = self.decoder(fused_featurs)

                    semantics = self.semantics_mlp(decoder_featurs)
                    density = self.density_mlp(decoder_featurs).reshape(-1)
                    pred_semantics = semantics.argmax(-1)
                    voting_semantics[index] = pred_semantics
                    voting_density[index] = density

                pred_semantics, indices = torch.mode(voting_semantics, dim=0)
                pred_density = voting_density.mean(dim=0)
            else:
                point_coors, tpv_featurs = self.query_point_feautres(
                    tpv_list=[tpv_hw, tpv_zh, tpv_wz],
                    tpv_scale=[1.0, 1.0, 1.0],
                    voxel_size=[0.4, 0.4, 0.4],
                    positions=center_coords
                )
                fused_featurs = self.ffpe(point_coors, tpv_featurs)

                semantics, pred_density = self.nerf_decoder(fused_featurs)
                pred_semantics = semantics.argmax(-1)

            non_empty_mask = (pred_density > self.density_threshold).reshape(-1)
            occ_pred[non_empty_mask] = pred_semantics[non_empty_mask]
            occ_preds.append(occ_pred)

        return occ_preds

    def loss(self, tpv_list, batch_rays_bundle, batch_data_samples):
        tpv_hw, tpv_zh, tpv_wz = tpv_list
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        losses = dict()
        for i, data_sample in enumerate(batch_data_samples):
            rays_bundle = batch_rays_bundle[i]
            H, W = data_sample.batch_input_shape[0], data_sample.batch_input_shape[1]
            for cam_num in range(rays_bundle.shape[0]):
                # get gt depth and semantics map

                gt_depth = data_sample.gt_maps['depth_maps'][cam_num].unsqueeze(0)
                gt_semantics = data_sample.gt_maps['semantics_maps'][cam_num].to(torch.int64)

                origins, directions, viewdirs, x, y = (rays_bundle[cam_num][:, :3],
                                                       rays_bundle[cam_num][:, 3:6],
                                                       rays_bundle[cam_num][:, 6:9],
                                                       rays_bundle[cam_num][9],
                                                       rays_bundle[cam_num][10])
                # if self.s3imloss_weight > 0:
                #     sem_map = torch.full(size=(H, W), fill_value=255, device=gt_semantics.device, dtype=torch.float32)

                if self.supervised_rgb:
                    gt_rgb = data_sample.gt_maps['rgb_maps'][cam_num].to(torch.float64)

                ray_indices, t_starts, t_ends = nerfacc.ray_marching(origins, viewdirs,
                                                                     near_plane=self.near_planes[cam_num],
                                                                     far_plane=self.far_planes[cam_num],
                                                                     render_step_size=self.render_step_size,
                                                                     stratified=True)
                distances = (t_starts + t_ends) / 2.0

                point_coors, tpv_featurs = self.query_point_feautres(
                    tpv_list=[tpv_hw, tpv_zh, tpv_wz],
                    tpv_scale=[1.0, 1.0, 1.0],
                    voxel_size=[0.4, 0.4, 0.4],
                    origins=origins,
                    viewdirs=viewdirs,
                    ray_indices=ray_indices,
                    t_starts=t_starts,
                    t_ends=t_ends
                )
                semantics, rgbs, density = self.nerf_decoder(point_coors, tpv_featurs, input_viewdirs=viewdirs[ray_indices])
                # rendering
                weights = nerfacc.render_weight_from_density(
                    t_starts,
                    t_ends,
                    density,
                    ray_indices=ray_indices,
                    n_rays=origins.shape[0],
                ).to(torch.float64)

                render_semantics = nerfacc.accumulate_along_rays(
                    weights,
                    ray_indices,
                    values=semantics,
                    n_rays=origins.shape[0])

                render_depths = nerfacc.accumulate_along_rays(
                    weights,
                    ray_indices,
                    values=distances,
                    n_rays=origins.shape[0],
                ).transpose(0, 1)
                loss_semantics = self.loss_semantics(render_semantics, gt_semantics)
                loss_depth = self.loss_depth(render_depths, gt_depth)
                if self.supervised_rgb:
                    render_rgbs = nerfacc.accumulate_along_rays(
                        weights,
                        ray_indices,
                        values=rgbs,
                        n_rays=origins.shape[0]
                    )
                    loss_rgb = F.smooth_l1_loss(render_rgbs, gt_rgb)
                    if 'loss_rgb' not in losses:
                        losses['loss_rgb'] = loss_rgb
                    else:
                        losses['loss_rgb'] += loss_rgb

                    if self.s3imloss_weight > 0:
                        loss_s3im = self.loss_s3im(render_rgbs, gt_rgb)
                        if 'loss_s3im' not in losses:
                            losses['loss_s3im'] = loss_s3im
                        else:
                            losses['loss_s3im'] += loss_s3im

                if 'loss_semantics' not in losses:
                    losses['loss_semantics'] = loss_semantics
                    losses['loss_depth'] = loss_depth
                    if self.distloss_weight > 0:
                        loss_distloss = self.distloss_weight * flatten_eff_distloss(weights.flatten().to(torch.float32),
                                                                                    distances.flatten(),
                                                                                    self.render_step_size,
                                                                                    ray_indices.flatten())
                        losses['loss_distloss'] = loss_distloss
                else:
                    losses['loss_semantics'] += loss_semantics
                    losses['loss_depth'] += loss_depth

        return losses
