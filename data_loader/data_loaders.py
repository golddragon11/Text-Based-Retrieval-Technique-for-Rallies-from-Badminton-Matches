from base import BaseDataLoader
from dataset.badminton_dataset import BadmintonShotDataset, RallyDataset
from dataset.textFeature import SentenceBERT
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from transformers import BertModel, BertTokenizer
import torch


class BadmintonShotDataLoader(BaseDataLoader):
    def __init__(self, dataset_kwargs, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = BadmintonShotDataset(**dataset_kwargs, training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class RallyDataLoader(BaseDataLoader):
    def __init__(self, dataset_kwargs, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        word_embedder = BertModel.from_pretrained('bert-base-uncased')    # dim=768
        word_embedder.eval()
        word_embedder = word_embedder.to('cuda')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.dataset = RallyDataset(**dataset_kwargs, word_embedding=word_embedder, tokenizer=tokenizer, training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
