import collections

from timm.models.vision_transformer import trunc_normal_
import torch.nn as nn
from functools import partial
import torch
from model.utils.layers import FusionBlock


class VideoTracknetFusion(nn.Module):
    def __init__(self, embed_dim=768, depth=1, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None,
                 use_cls_token=False,
                 ):
        super().__init__()

        self.embed_dim = embed_dim

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.masking_token = nn.Parameter(torch.zeros(embed_dim))

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            FusionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
            )
            for i in range(depth)])

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.masking_token, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

    def forward(self, video=None, tracknet=None):
        # Concatenate tokens
        data = [video, tracknet]
        tokens = [x['tokens'] for x in data if x is not None]
        tokens = torch.cat(tokens, dim=1)

        # Concatenate attention masks
        attention_mask = [x['attention_mask'] for x in data if x is not None]
        attention_mask = torch.cat(attention_mask, dim=1)

        for block in self.blocks:
            tokens = block(tokens, attention_mask=attention_mask)

        return tokens
