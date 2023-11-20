import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS

from torch_efficient_distloss import flatten_eff_distloss
import nerfacc


@MODELS.register_module()
class TriNeOccHead(BaseModule):

    def __init__(self,
                 tpv_h,
                 tpv_w,
                 tpv_z,
                 scene_aabb,
                 near_planes,
                 far_planes,
                 aug_infer=False,
                 voxel_size=0.4,
                 render_step_size=0.2,
                 resample_number=96,
                 density_threshold=0.05,
                 num_classes=18,
                 in_dims=64,
                 position_dim=3,
                 hidden_dims=128,
                 out_dims=None,
                 scale_h=2,
                 scale_w=2,
                 scale_z=2,
                 ignore_index=0,
                 distloss_weight=0,
                 supervised_rgb=True,
                 nerf_decoder=None,
                 pro_nerf_decoder=None,
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
        self.aug_infer = aug_infer
        self.position_dim = position_dim
        self.distloss_weight = distloss_weight
        self.density_threshold = density_threshold
        self.voxel_size = voxel_size
        self.scene_aabb = torch.tensor(scene_aabb, device='cuda')
        self.render_step_size = render_step_size
        self.supervised_rgb = supervised_rgb
        self.resample_number = resample_number

        self.min_bound = self.scene_aabb[:3]
        self.max_bound = self.scene_aabb[3:]
        self.near_planes = near_planes
        self.far_planes = far_planes

        out_dims = in_dims if out_dims is None else out_dims
        self.in_dims = in_dims

        if pro_nerf_decoder is not None:
            self.pro_nerf_decoder = MODELS.build(pro_nerf_decoder)
        else:
            self.pro_nerf_decoder = None
        self.nerf_decoder = MODELS.build(nerf_decoder)
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
        point_coors[:, 0] = (point_coors[:, 0] - self.min_bound[0]) / point_coors.new_tensor(voxel_size[0])
        point_coors[:, 1] = (point_coors[:, 1] - self.min_bound[1]) / point_coors.new_tensor(voxel_size[1])
        point_coors[:, 2] = (point_coors[:, 2] - self.min_bound[2]) / point_coors.new_tensor(voxel_size[2])

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
            semantics_logits, pred_density = self.nerf_decoder(point_coors, tpv_featurs)
            seg_pred = semantics_logits.argmax(-1)
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
                    semantics_logits, pred_density = self.nerf_decoder(point_coors, tpv_featurs)

                    pred_semantics = semantics_logits.argmax(-1)
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
                semantics_logits, pred_density = self.nerf_decoder(point_coors, tpv_featurs)
                pred_semantics = semantics_logits.argmax(-1)

            non_empty_mask = (pred_density > self.density_threshold).reshape(-1)
            occ_pred[non_empty_mask] = pred_semantics[non_empty_mask]
            occ_preds.append(occ_pred)

        return occ_preds

    def rendering(self,
                  ray_indices,
                  t_starts,
                  t_ends,
                  n_rays,
                  nerf_decoder,
                  pts,
                  pts_tpv_features,
                  input_viewdirs=None,
                  resample_number=None
                  ):
        render_result = {}

        if input_viewdirs is not None:
            semantics, rgbs, density = nerf_decoder(pts, pts_tpv_features, input_viewdirs=input_viewdirs)
        else:
            semantics, density = nerf_decoder(pts, pts_tpv_features)

        weights = nerfacc.render_weight_from_density(
            t_starts,
            t_ends,
            density,
            ray_indices=ray_indices,
            n_rays=n_rays,
        ).to(torch.float64)

        render_semantics = nerfacc.accumulate_along_rays(
            weights,
            ray_indices,
            values=semantics,
            n_rays=n_rays)

        render_depths = nerfacc.accumulate_along_rays(
            weights,
            ray_indices,
            values=(t_starts + t_ends) / 2.0,
            n_rays=n_rays,
        ).transpose(0, 1)
        render_result['weights'] = weights
        render_result['render_semantics'] = render_semantics
        render_result['render_depths'] = render_depths

        if input_viewdirs is not None:
            render_rgbs = nerfacc.accumulate_along_rays(
                weights,
                ray_indices,
                values=rgbs,
                n_rays=n_rays
            )
            render_result['render_rgbs'] = render_rgbs

        if resample_number is not None:
            packed_info = nerfacc.pack_info(ray_indices, n_rays=n_rays)
            packed_info, t_starts, t_ends = nerfacc.ray_resampling(
                packed_info,
                t_starts.to(torch.float64),
                t_ends.to(torch.float64),
                weights.flatten(),
                n_samples=resample_number
            )
            ray_indices = nerfacc.unpack_info(packed_info, t_starts.shape[0])
            render_result['ray_indices'] = ray_indices
            render_result['t_starts'] = t_starts
            render_result['t_ends'] = t_ends

        return render_result

    def compute_loss(self, render_results):
        losses = {}
        losses['loss_semantics'] = self.loss_semantics(render_results['render_semantics'],
                                                       render_results['gt_semantics'])
        losses['loss_depth'] = self.loss_depth(render_results['render_depths'], render_results['gt_depths'])
        # supervised rgb
        if self.supervised_rgb:
            losses['loss_rgb'] = F.smooth_l1_loss(render_results['render_rgbs'], render_results['gt_rgbs'])

        # distloss
        if self.distloss_weight > 0:
            losses['loss_dist'] = self.distloss_weight * flatten_eff_distloss(
                render_results['weights'].flatten().to(torch.float32),
                ((render_results['t_starts'] + render_results['t_ends']) / 2.0).flatten(),
                (render_results['t_ends'] - render_results['t_starts']).flatten(),
                render_results['ray_indices'].flatten())

        return losses

    def loss(self, tpv_list, batch_rays_bundle, batch_data_samples):
        tpv_hw, tpv_zh, tpv_wz = tpv_list
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        losses = dict()
        for i, data_sample in enumerate(batch_data_samples):
            rays_bundle = batch_rays_bundle[i]
            for cam_num in range(rays_bundle.shape[0]):
                render_results = dict()
                # get gt depth and semantics map
                gt_depths = data_sample.gt_maps['depth_maps'][cam_num].unsqueeze(0)
                gt_semantics = data_sample.gt_maps['semantics_maps'][cam_num].to(torch.int64)
                render_results['gt_depths'] = gt_depths
                render_results['gt_semantics'] = gt_semantics
                # get sampling points (ego coord)
                origins, directions, viewdirs = rays_bundle[cam_num][:, :3], rays_bundle[cam_num][:, 3:6], rays_bundle[
                                                                                                               cam_num][
                                                                                                           :, 6:9]
                if self.supervised_rgb:
                    gt_rgbs = data_sample.gt_maps['rgb_maps'][cam_num]
                    render_results['gt_rgbs'] = gt_rgbs

                ray_indices, t_starts, t_ends = nerfacc.ray_marching(origins, viewdirs,
                                                                     near_plane=self.near_planes[cam_num],
                                                                     far_plane=self.far_planes[cam_num],
                                                                     render_step_size=self.render_step_size,
                                                                     stratified=True)
                render_results['ray_indices'] = ray_indices
                render_results['t_starts'] = t_starts
                render_results['t_ends'] = t_ends
                # query point features
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
                render_results.update(
                    self.rendering(
                        ray_indices,
                        t_starts,
                        t_ends,
                        origins.shape[0],
                        self.nerf_decoder,
                        point_coors,
                        tpv_featurs,
                        input_viewdirs=(viewdirs[ray_indices] if self.supervised_rgb else None)
                    )
                )
                losses_single = self.compute_loss(render_results)
                for key in losses_single:
                    if key in losses and key != 'loss_dist':
                        losses[key] = losses[key] + losses_single[key]
                    else:
                        losses[key] = losses_single[key]

        return losses
