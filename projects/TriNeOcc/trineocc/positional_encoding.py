import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


class FrequencyEmbedder(BaseModule):
    def __init__(self,
                 input_dim,
                 max_freq_log2,
                 N_freqs,
                 log_sampling=True,
                 include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


class SELayer(BaseModule):
    def __init__(self, in_channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Linear(in_channels, in_channels)
        self.act1 = act_layer()
        self.conv_expand = nn.Linear(in_channels, in_channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


@MODELS.register_module()
class TPVFrequencyFeaturePE(BaseModule):

    def __init__(self,
                 use_pts_freq_embed,
                 use_view_freq_embed,
                 position_dim,
                 pts_max_freq_log2,
                 view_max_freq_log2,
                 in_dims):
        super().__init__()
        self.use_pts_freq_embed = use_pts_freq_embed
        self.use_view_freq_embed = use_view_freq_embed

        if use_view_freq_embed:
            self.view_freq_embed = FrequencyEmbedder(input_dim=position_dim, max_freq_log2=view_max_freq_log2 - 1,
                                                N_freqs=view_max_freq_log2)

        if use_pts_freq_embed:
            self.pts_freq_embed = FrequencyEmbedder(input_dim=position_dim, max_freq_log2=pts_max_freq_log2 - 1,
                                                N_freqs=pts_max_freq_log2)
            position_dim = self.pts_freq_embed.out_dim

        self.position_encoder = nn.Sequential(
            nn.Linear(position_dim, in_dims),
            nn.Softplus(),
            nn.Linear(in_dims, in_dims)
        )
        self.feature_embed = SELayer(in_dims)

    def forward(self, point_coors, tpv_featurs):
        if self.use_pts_freq_embed:
            pts_features = self.pts_freq_embed(point_coors)
        else:
            pts_features = point_coors
        pts_features = self.position_encoder(pts_features)
        fused_featurs =  tpv_featurs + self.feature_embed(pts_features, tpv_featurs)
        return fused_featurs
