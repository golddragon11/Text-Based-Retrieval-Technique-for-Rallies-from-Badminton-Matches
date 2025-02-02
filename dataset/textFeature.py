import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


class SimCSE(nn.Module):
    def __init__(self, model=AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased"), device='cuda'):
        super(SimCSE, self).__init__()
        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        self.model = model
        self.model.to(self.device)

    def forward(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs.to(self.device)
        # Get the embeddings
        with torch.no_grad():
            embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        return embeddings


class SentenceBERT(nn.Module):
    def __init__(self, model=SentenceTransformer('all-mpnet-base-v2'), device='cuda'):
        super(SentenceBERT, self).__init__()
        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model = model
        self.model.to(self.device)

    def forward(self, inputs):
        # Get the embeddings
        with torch.no_grad():
            embeddings = self.model.encode(inputs)[np.newaxis, :]
            embeddings = torch.from_numpy(embeddings)
        return embeddings
