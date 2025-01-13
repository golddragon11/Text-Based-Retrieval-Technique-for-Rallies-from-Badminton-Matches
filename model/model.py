import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from model.multiModalTransformer import VideoTracknetFusion
from model.utils.layers import get_projection, FusionBlock

from functools import partial


class ShotClassificationModel(nn.Module):
    """
    Input Layer -> Encoder Layer -> Pooling Layer -> Dense Layer -> Softmax
    """
    def __init__(self, fusion_params, n_classes, n_tokens, video_embed_dim, tracknet_embed_dim, token_dim, token_projection):
        super(ShotClassificationModel, self).__init__()
        self.fusion = VideoTracknetFusion(**fusion_params)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.video_token_proj = get_projection(video_embed_dim, token_dim, token_projection)
        self.tracknet_token_proj = get_projection(tracknet_embed_dim, token_dim, token_projection)

        self.video_norm_layer = nn.LayerNorm(token_dim, eps=1e-6)
        self.tracknet_norm_layer = nn.LayerNorm(token_dim, eps=1e-6)

        self.classifier = nn.Sequential(
            get_projection(token_dim, n_classes, 'gated'),
            nn.LayerNorm(n_classes, eps=1e-6),
            nn.AvgPool2d((n_tokens, 1))
        )

    def forward(self, data):
        output = {}
        video = self.extract_video_tokens(data['video_token'], data['video_mask'])
        tracknet = self.extract_tracknet_tokens(data['tracknet_token'], data['tracknet_mask'])

        tokens = self.fusion(video=video, tracknet=tracknet)

        output['raw'] = self.classifier(tokens).squeeze(1)
        output['prob'] = self.softmax(output['raw'])
        output['log_prob'] = self.log_softmax(output['raw'])

        return output

    def extract_video_tokens(self, video, attention_mask):
        x = self.video_token_proj(video)
        x = self.video_norm_layer(x)

        return {'tokens': x, 'attention_mask': attention_mask}

    def extract_tracknet_tokens(self, tracknet, attention_mask):
        tracknet = self.tracknet_token_proj(tracknet)
        tracknet = self.tracknet_norm_layer(tracknet)

        return {'tokens': tracknet, 'attention_mask': attention_mask}


class TextBasedDataRetrievalModel(nn.Module):
    def __init__(self, text_embed_dim=768, training=True):
        super(TextBasedDataRetrievalModel, self).__init__()
        self.training = training

        self.text_transformer_encoder = nn.Sequential(*[
            FusionBlock(
                dim=text_embed_dim, num_heads=256, mlp_ratio=2, qkv_bias=True, drop=0.1,
                attn_drop=0, drop_path=0, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.Tanh,
            )
            for i in range(3)])
        self.text_encoder = nn.Sequential(
            get_projection(text_embed_dim, text_embed_dim, 'gated'),
            nn.LayerNorm(text_embed_dim, eps=1e-6),
            nn.AvgPool2d((25, 1))
        )

    def forward(self, data):
        output = {}

        text = data['text_token']
        if text is not None:
            for block in self.text_transformer_encoder:
                text = block(text, attention_mask=data['text_mask'])
            text = self.text_encoder(text).squeeze(1)

        if data['description_token'] is not None:
            description = []
            for i in range(data['description_token'].shape[1]):
                token = data['description_token'][:, i].squeeze(1)
                for block in self.text_transformer_encoder:
                    token = block(token, attention_mask=data['description_mask'])
                description.append(self.text_encoder(token).squeeze(1))
            description = torch.stack(description, dim=1)
            description = nn.AvgPool2d((8, 1))(description).squeeze(1)
        else:
            description = None

        output['text_embed'] = text
        output['data_embed'] = description

        return output
