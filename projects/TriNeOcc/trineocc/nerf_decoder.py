import numpy as np

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SemNerfDecoder(BaseModule):
    def __init__(self,
                 depth=8,
                 rgb_depth=1,
                 sem_depth=1,
                 hidden_dim=256,
                 num_classes=17,
                 input_ch=3,
                 input_ch_viewdirs=3,
                 skips=[4],
                 rgb_branch=False,
                 use_viewdirs=False,
                 ffpe=None):
        '''
        :param depth: network depth
        :param hidden_dim: network width
        :param input_ch: input channels for encodings of (x, y, z)
        :param input_ch_viewdirs: input channels for encodings of view directions
        :param skips: skip connection in network
        :param use_viewdirs: if True, will use the view directions as input
        '''
        super().__init__()

        self.input_ch = input_ch
        self.input_ch_viewdirs = input_ch_viewdirs
        self.use_viewdirs = use_viewdirs
        self.skips = skips

        self.decoder_layers = []
        dim = self.input_ch
        for i in range(depth):
            self.decoder_layers.append(
                nn.Sequential(nn.Linear(dim, hidden_dim), nn.Softplus())
            )
            dim = hidden_dim
            if i in self.skips and i != (depth - 1):  # skip connection after i^th layer
                dim += input_ch
        self.decoder_layers = nn.ModuleList(self.decoder_layers)

        density_layers = [nn.Linear(dim, 1), nn.Softplus()]  # sigma must be positive
        self.density_layers = nn.Sequential(*density_layers)

        # sem
        semantics_layers = []
        base_remap_layers = [nn.Linear(dim, hidden_dim), ]
        self.base_remap_layers = nn.Sequential(*base_remap_layers)

        sem_hidden_dim = hidden_dim
        for i in range(sem_depth):
            semantics_layers.append(nn.Linear(dim, sem_hidden_dim // 2))
            semantics_layers.append(nn.Softplus())
            sem_hidden_dim = sem_hidden_dim // 2
            dim = sem_hidden_dim
        semantics_layers.append(nn.Linear(dim, num_classes))
        self.semantics_layers = nn.Sequential(*semantics_layers)

        if rgb_branch:
            rgb_layers = []
            rgb_dim = hidden_dim + self.input_ch_viewdirs
            rgb_hidden_dim = hidden_dim
            for i in range(rgb_depth):
                rgb_layers.append(nn.Linear(rgb_dim, rgb_hidden_dim // 2))
                rgb_layers.append(nn.Softplus())
                rgb_hidden_dim = rgb_hidden_dim // 2
                rgb_dim = rgb_hidden_dim
            rgb_layers.append(nn.Linear(rgb_dim, 3))
            rgb_layers.append(nn.Sigmoid())  # rgb values are normalized to [0, 1]
            self.rgb_layers = nn.Sequential(*rgb_layers)

        self.ffpe = MODELS.build(ffpe)

    def forward(self, pts, pts_tpv_features, input_viewdirs=None):
        '''
        :param input: [..., input_ch+input_ch_viewdirs]
        :return [..., 4]
        '''
        # positional encoding
        pts_fused_features = self.ffpe(pts, pts_tpv_features)
        if input_viewdirs is not None:
            viewdirs_features = self.ffpe.view_freq_embed(input_viewdirs)

        base = self.decoder_layers[0](pts_fused_features)
        for i in range(len(self.decoder_layers) - 1):
            if i in self.skips:
                base = torch.cat((pts_fused_features, base), dim=-1)
            base = self.decoder_layers[i + 1](base)

        density = self.density_layers(base)
        base_remap = self.base_remap_layers(base)

        semantics = self.semantics_layers(base_remap)
        if input_viewdirs is not None:
            rgbs = self.rgb_layers(torch.cat((base_remap, viewdirs_features), dim=-1))
            return semantics, rgbs, density

        return semantics, density
