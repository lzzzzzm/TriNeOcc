import mmcv
from mmengine.dataset import BaseDataset
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet3d.registry import HOOKS
from typing import Optional, Sequence

from mmdet3d.visualization import Det3DLocalVisualizer

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


@HOOKS.register_module()
class CustomHook(Hook):

    def __init__(self,
                 loss_hook_enable=False,
                 pts_seg_enable=False,
                 render_enable=False,
                 enable_epoch: int = 6,
                 loss_weight: float = 0.1):
        self.loss_hook_enable = loss_hook_enable
        self.render_enable = render_enable
        self.pts_seg_enable = pts_seg_enable
        self.enable_epoch = enable_epoch
        self.loss_weight = loss_weight

    def before_test_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch) -> None:
        if self.render_enable:
            index_map = {0: 'CAM_FRONT', 1: 'CAM_LEFT', 2: 'CAM_RIGHT', 3: "CAM_BACK", 4: "CAM_BACKLEFT",
                         5: "CAM_BACKRIGHT"}
            imgs = data_batch['inputs']['img'][0].permute(0, 2, 3, 1).cpu().numpy()
            for index in range(len(imgs)):
                mmcv.imwrite(imgs[index], 'vis/input_img/{}.jpg'.format(index_map[index]))
            runner.logger.info('write ori img in vis dic')

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs) -> None:
        if self.pts_seg_enable:
            vis = Det3DLocalVisualizer()
            points = data_batch['inputs']['points'][0]
            pred_pts_seg = outputs[0].pred_pts_seg
            gt_pts_seg = data_batch['data_samples'][0].gt_pts_seg
            vis._draw_pts_pred_gt_sem_seg(points, pred_pts_seg, gt_pts_seg, palette)
            vis.show()
            vis.o3d_vis.destroy_window()
        if self.render_enable:
            index_map = {0: 'CAM_FRONT', 1: 'CAM_LEFT', 2: 'CAM_RIGHT', 3: "CAM_BACK", 4: "CAM_BACKLEFT",
                         5: "CAM_BACKRIGHT"}
            render_maps = outputs[0].render_maps
            depth_maps, rgb_maps, semantics_maps = render_maps['depth_maps'], render_maps['rgb_maps'], render_maps[
                'semantics_maps']
            for i in range(len(depth_maps)):
                depth_map, rgb_map, sem_map = depth_maps[i], rgb_maps[i], semantics_maps[i]
                mmcv.imwrite(depth_map, 'vis/depth/{}.jpg'.format(index_map[i]))
                mmcv.imwrite(rgb_map, 'vis/rgb/{}.jpg'.format(index_map[i]))
                mmcv.imwrite(sem_map, 'vis/sem/{}.jpg'.format(index_map[i]))
            runner.logger.info('write render result in vis dic')

    def before_train_epoch(self, runner):
        if self.loss_hook_enable:
            epoch = runner.epoch
            model = runner.model
            if is_model_wrapper(model):
                model = model.module
            if epoch == self.enable_epoch:
                runner.logger.info('Enable Distloss')
                model.decode_head.distloss_weight = self.loss_weight
