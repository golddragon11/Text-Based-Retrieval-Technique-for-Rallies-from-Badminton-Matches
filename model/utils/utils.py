import torch
import numpy as np


def normalize_embeddings(a, eps=1e-8):
    a_n = a.norm(dim=1)[:, np.newaxis]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a = normalize_embeddings(a, eps)
    b = normalize_embeddings(b, eps)

    sim_mt = torch.mm(a, b.transpose(0, 1))

    return sim_mt
