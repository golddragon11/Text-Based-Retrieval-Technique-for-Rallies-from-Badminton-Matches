from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pickle
import random
import numpy as np


class BadmintonShotDataset(Dataset):
    def __init__(self, data_path, n_classes, n_video_tokens=48, n_tracknet_tokens=512, training=True):
        self.data = pickle.load(open(data_path, 'rb'))
        self.n_classes = n_classes
        self.n_video_tokens = n_video_tokens
        self.n_tracknet_tokens = n_tracknet_tokens
        self.training = training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id_ = self.data[idx]['id']
        cls = self.data[idx]['shot_type']

        video_embedding = self.data[idx]['videoFeature']
        video_token, video_mask = self.video_tokenization(video_embedding, self.n_video_tokens)

        tracknet = torch.from_numpy(self.data[idx]['tracknet'])
        tracknet_token, tracknet_mask = self.tracknet_tokenization(tracknet, self.n_tracknet_tokens)

        dataset = "BadmintonShot"

        # Create one-hot encoding for data[idx]['class']
        class_one_hot = torch.zeros(self.n_classes)
        class_one_hot[cls] = 1
        class_one_hot = class_one_hot.float()

        return {'video_token': video_token, 'tracknet_token': tracknet_token, 'video_mask': video_mask, 'tracknet_mask': tracknet_mask, 'label': class_one_hot,
                'meta': {'id': id_, 'dataset': dataset, 'class': cls}}

    def video_tokenization(self, data, n_tokens):
        if data.shape[0] <= n_tokens:
            cur_n_tokens, _ = data.shape
            token = torch.zeros(n_tokens, data.shape[-1])
            # token[:cur_n_tokens] = F.normalize(data[:cur_n_tokens], dim=1)
            token[:cur_n_tokens] = data[:cur_n_tokens]
            attention_mask = torch.zeros(n_tokens)
            attention_mask[:cur_n_tokens] = 1
        else:
            # token = torch.nn.functional.interpolate(
            #             data.permute(1, 0).unsqueeze(0),
            #             size=n_tokens,
            #             mode='nearest').squeeze(0).permute(1, 0)
            token = data[:n_tokens]
            # token = F.normalize(token, dim=1)
            attention_mask = torch.ones(n_tokens)
        return token.float(), attention_mask.float()

    def tracknet_tokenization(self, data, n_tokens):
        token = torch.zeros(n_tokens, 12)
        if data.shape[0] <= n_tokens:
            cur_n_tokens, _ = data.shape
            # token[:cur_n_tokens] = F.normalize(data[:cur_n_tokens, 1:], dim=1)
            token[:cur_n_tokens] = data[:cur_n_tokens, 1:]
            attention_mask = torch.zeros(n_tokens)
            attention_mask[:cur_n_tokens] = data[:cur_n_tokens, 0]
        else:
            # token = F.normalize(data[:n_tokens, 1:], dim=1)
            token = data[:n_tokens, 1:]
            attention_mask = data[:n_tokens, 0]
        token += 1e-5
        return token.float(), attention_mask.float()


class RallyDataset(Dataset):
    def __init__(self, data_path, n_classes, word_embedding=None, tokenizer=None, max_words=25, training=True) -> None:
        super().__init__()
        self.data = pickle.load(open(data_path, 'rb'))
        self.n_classes = n_classes
        self.word_embedding = word_embedding
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.training = training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id_ = self.data[idx]['id']
        service_type = self.data[idx]['service_type']
        last_shot_type = self.data[idx]['last_shot_type']
        rally_length = self.data[idx]['rally_length']
        most_frequent = self.data[idx]['most_frequent']
        shot_counts = self.data[idx]['shot_counts']
        description = self.data[idx]['description']
        caption = self.data[idx]['caption']
        cls = np.array(self.data[idx]['class'])

        if cls.size < 5:
            cls = np.pad(cls, (0, 5 - cls.size), 'constant', constant_values=-1)

        dataset = "Rally"

        x = np.concatenate([[service_type], [last_shot_type], [rally_length], [most_frequent], shot_counts]).astype(np.float32)
        x = np.expand_dims(x, axis=0)
        selected_caption = caption
        eval_class = self.data[idx]['eval_class']
        with torch.no_grad():
            text_token, text_mask = self.create_text_tokens_BERT(selected_caption)
            description_token, description_mask = self.create_text_tokens_BERT(description)

        return {'data_token': x, 'text_token': text_token, 'text_mask': text_mask, 'description_token': description_token, 'description_mask': description_mask,
                'meta': {'id': id_, 'dataset': dataset, 'class': cls, 'eval_class': eval_class}}

    def create_text_tokens_BERT(self, caption):
        encoding = self.tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_words, add_special_tokens=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        with torch.no_grad():
            outputs = self.word_embedding(input_ids, attention_mask=attention_mask)
            text_token = outputs.last_hidden_state  # This contains the embeddings
        text_token = text_token.squeeze(0)
        text_mask = np.ones(self.max_words, dtype=np.float32)
        text_mask = torch.from_numpy(text_mask).float()

        return text_token, text_mask
