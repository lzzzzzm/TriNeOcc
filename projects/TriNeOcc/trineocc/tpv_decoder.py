import numpy as np

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS

@MODELS.register_module()
class TPVDecoder(BaseModule):

    def __init__(self,
                 tpv_h=50,
                 tpv_w=50,
                 tpv_z=4,
                 in_dims=128,
                 out_dims=128):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.upsample_layer = int(np.math.log2(200//tpv_h))*3
        ori_in_dims = in_dims
        upsample = []
        for i in range(self.upsample_layer):
            if i%3==0 and i!=0:
                in_dims = in_dims//2

            tpv_upsample = nn.Sequential(
                nn.ConvTranspose2d(in_dims, in_dims//2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(in_dims // 2),
                nn.ReLU())
            upsample.append(tpv_upsample)
        self.upsample = nn.Sequential(*upsample)

        in_dims = ori_in_dims//2
        channel_increase = []
        for i in range(self.upsample_layer):
            if i%3==0 and i!=0:
                in_dims = in_dims//2
            increase = nn.Sequential(
                nn.Conv2d(in_dims, out_dims, kernel_size=1),
                nn.BatchNorm2d(out_dims),
                nn.ReLU())
            channel_increase.append(increase)
        self.channel_increase = nn.Sequential(*channel_increase)


    def forward(self, tpv_list):
        tpv_hw, tpv_zh, tpv_wz = tpv_list
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        tpv_plane_output = [tpv_hw, tpv_zh, tpv_wz]
        for index in range(self.upsample_layer//3):
            tpv_list = tpv_plane_output[index*3:(index+1)*3]
            for i, tpv_plane in enumerate(tpv_list):
                upsample_tpv_plane = self.upsample[index*3+i](tpv_plane)
                tpv_plane_output.append(upsample_tpv_plane)

        tpv_increase_output = tpv_plane_output[:3]
        for index in range(1, self.upsample_layer//3+1):
            tpv_list = tpv_plane_output[index * 3:(index + 1) * 3]
            for i, tpv_plane in enumerate(tpv_list):
                increase_output = self.channel_increase[(index-1)*3+i](tpv_plane)
                tpv_increase_output.append(increase_output)

        return tpv_increase_output
