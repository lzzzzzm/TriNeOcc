import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS

import nerfacc

@MODELS.register_module()
class TriNeOccHead(BaseModule):

    def __init__(self,
                 tpv_h,
                 tpv_w,
                 tpv_z,
                 voxel_size=0.4,
                 scene_aabb=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                 render_step_size=0.2,
                 stratified=True,
                 num_classes=18,
                 in_dims=64,
                 hidden_dims=128,
                 out_dims=None,
                 scale_h=2,
                 scale_w=2,
                 scale_z=2,
                 ignore_index=0,
                 loss_semantics=None,
                 loss_depth=None):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        # render params
        self.voxel_size = voxel_size
        self.scene_aabb = torch.tensor(scene_aabb, device='cuda')
        self.render_step_size = render_step_size
        self.stratified = stratified

        self.min_bound = self.scene_aabb[:3]
        self.max_bound = self.scene_aabb[3:]

        out_dims = in_dims if out_dims is None else out_dims
        self.in_dims = in_dims
        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims))

        self.semantics_mlp = nn.Sequential(
            nn.Linear(out_dims, out_dims*2),
            nn.Softplus(),
            nn.Linear(out_dims*2, num_classes - 1)
        )
        self.density_mlp = nn.Sequential(
            nn.Linear(out_dims, out_dims*2),
            nn.Softplus(),
            nn.Linear(out_dims*2, 1),
            nn.Softplus()
        )

        self.loss_semantics = MODELS.build(loss_semantics)
        self.loss_depth = MODELS.build(loss_depth)
        self.ignore_index = ignore_index


    def forward(self, tpv_list, points=None):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw,
                size=(self.tpv_h * self.scale_h, self.tpv_w * self.scale_w),
                mode='bilinear')
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh,
                size=(self.tpv_z * self.scale_z, self.tpv_h * self.scale_h),
                mode='bilinear')
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz,
                size=(self.tpv_w * self.scale_w, self.tpv_z * self.scale_z),
                mode='bilinear')

        logits = ()
        return logits

    def predict(self, tpv_list, batch_data_samples):
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

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw,
                size=(self.tpv_h * self.scale_h, self.tpv_w * self.scale_w),
                mode='bilinear')
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh,
                size=(self.tpv_z * self.scale_z, self.tpv_h * self.scale_h),
                mode='bilinear')
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz,
                size=(self.tpv_w * self.scale_w, self.tpv_z * self.scale_z),
                mode='bilinear')

        logits = []

        return logits

    def loss(self, tpv_list, batch_rays_bundle, batch_data_samples):
        tpv_hw, tpv_zh, tpv_wz = tpv_list
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw,
                size=(self.tpv_h * self.scale_h, self.tpv_w * self.scale_w),
                mode='bilinear')
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh,
                size=(self.tpv_z * self.scale_z, self.tpv_h * self.scale_h),
                mode='bilinear')
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz,
                size=(self.tpv_w * self.scale_w, self.tpv_z * self.scale_z),
                mode='bilinear')

        losses = dict()
        for i, data_sample in enumerate(batch_data_samples):
            rays_bundle = batch_rays_bundle[i]
            for cam_num in range(rays_bundle.shape[0]):
                # get gt depth and semantics map
                gt_depth = data_sample.gt_maps['depth_maps'][cam_num].unsqueeze(0)
                gt_semantics = data_sample.gt_maps['semantics_maps'][cam_num].to(torch.int64)
                # get sampling points (ego coord)
                origins, directions, viewdirs = rays_bundle[cam_num][:, :3], rays_bundle[cam_num][:, 3:6], rays_bundle[cam_num][:, 6:]
                ray_indices, t_starts, t_ends = nerfacc.ray_marching(origins, viewdirs, scene_aabb=self.scene_aabb,
                                                                     render_step_size=self.render_step_size, stratified=self.stratified)
                t_origins = origins[ray_indices]  # (n_samples, 3)
                t_dirs = viewdirs[ray_indices]  # (n_samples, 3)
                positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
                point_clamp = torch.clamp(positions, self.min_bound, self.max_bound + 1e-6)
                point_coors = torch.floor(
                    (point_clamp - self.min_bound) /
                    point_clamp.new_tensor(self.voxel_size))

                point_coors = point_coors.reshape(1, 1, -1, 3)
                point_coors[
                    ...,
                    0] = point_coors[..., 0] / (self.tpv_w * self.scale_w) * 2 - 1
                point_coors[
                    ...,
                    1] = point_coors[..., 1] / (self.tpv_h * self.scale_h) * 2 - 1
                point_coors[
                    ...,
                    2] = point_coors[..., 2] / (self.tpv_z * self.scale_z) * 2 - 1

                sample_loc = point_coors[..., [0, 1]]
                tpv_hw_pts = F.grid_sample(
                    tpv_hw[i:i + 1], sample_loc, align_corners=False)
                sample_loc = point_coors[..., [1, 2]]
                tpv_zh_pts = F.grid_sample(
                    tpv_zh[i:i + 1], sample_loc, align_corners=False)
                sample_loc = point_coors[..., [2, 0]]
                tpv_wz_pts = F.grid_sample(
                    tpv_wz[i:i + 1], sample_loc, align_corners=False)

                tpv_featurs = (tpv_hw_pts + tpv_zh_pts + tpv_wz_pts).squeeze(0).squeeze(1).transpose(0, 1)
                tpv_featurs = self.decoder(tpv_featurs)

                semantics = self.semantics_mlp(tpv_featurs)
                density = self.density_mlp(tpv_featurs)
                # rendering
                weights = nerfacc.render_weight_from_density(
                    t_starts,
                    t_ends,
                    density,
                    ray_indices=ray_indices,
                    n_rays=origins.shape[0],
                ).to(torch.float64)
                semantics = nerfacc.accumulate_along_rays(weights, ray_indices, values=semantics, n_rays=origins.shape[0])
                opacities = nerfacc.accumulate_along_rays(weights, ray_indices, values=None, n_rays=origins.shape[0])
                depths = nerfacc.accumulate_along_rays(
                    weights,
                    ray_indices,
                    values=(t_starts + t_ends) / 2.0,
                    n_rays=origins.shape[0],
                ).transpose(0, 1)
                loss_semantics = self.loss_semantics(semantics, gt_semantics)
                loss_depth = self.loss_depth(depths, gt_depth)
                if 'loss_semantics' not in losses:
                    losses['loss_semantics'] = loss_semantics
                    losses['loss_depth'] = loss_depth
                else:
                    losses['loss_semantics'] += loss_semantics
                    losses['loss_depth'] += loss_depth

        return losses
