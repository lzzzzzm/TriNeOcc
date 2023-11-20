from typing import Optional, Union

import mmcv
import torch
from torch import nn

from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.structures import PointData

@MODELS.register_module()
class TriNeOcc(Base3DSegmentor):

    def __init__(self,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 backbone=None,
                 neck=None,
                 encoder=None,
                 decode_head=None,
                 predict_task='render'):

        super().__init__(data_preprocessor=data_preprocessor)

        self.predict_task = predict_task

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.encoder = MODELS.build(encoder)
        self.decode_head = MODELS.build(decode_head)

    def extract_feat(self, img):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img_feats = self.backbone(img)

        if hasattr(self, 'neck'):
            img_feats = self.neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped

    def _forward(self, batch_inputs, batch_data_samples):
        pass

    def loss(self, batch_inputs: dict,
             batch_data_samples: SampleList) -> SampleList:
        img_feats = self.extract_feat(batch_inputs['imgs'])
        queries = self.encoder(img_feats, batch_data_samples)
        losses = self.decode_head.loss(queries, batch_inputs['rays_bundle'], batch_data_samples)
        return losses

    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Forward predict function."""
        img_feats = self.extract_feat(batch_inputs['imgs'])
        tpv_queries = self.encoder(img_feats, batch_data_samples)
        if self.predict_task == 'occ':
            occ_preds = self.decode_head.predict_occ(tpv_queries, batch_data_samples)
            for i in range(len(occ_preds)):
                occ_pred = occ_preds[i]
                batch_data_samples[i].set_data({
                    'pred_occ_seg':
                        PointData(**{'pts_semantic_occ': occ_pred})
                })
        elif self.predict_task == 'seg':
            seg_preds = self.decode_head.predict_seg(tpv_queries, batch_inputs['points'], batch_data_samples)
            for i in range(len(seg_preds)):
                seg_pred = seg_preds[i]
                batch_data_samples[i].set_data({
                    'pred_pts_seg':
                        PointData(**{'pts_semantic_mask': seg_pred})
                })
        elif self.predict_task == 'render':
            render_maps = self.decode_head.render_img(tpv_queries, batch_inputs['rays_bundle'], batch_data_samples)
            for i in range(len(batch_data_samples)):
                batch_data_samples[i].set_data({
                    'render_maps':render_maps,
                    'pred_occ_seg':PointData(**{'pts_semantic_occ': [None]})
                })

        return batch_data_samples

    def aug_test(self, batch_inputs, batch_data_samples):
        pass

    def encode_decode(self, batch_inputs: dict,
                      batch_data_samples: SampleList) -> SampleList:
        pass
