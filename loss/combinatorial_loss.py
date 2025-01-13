import torch
import torch.nn as nn

from loss.contrastive_losses import NormSoftmaxLoss, MMS_Loss
from model.utils.utils import sim_matrix


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.NLL = torch.nn.NLLLoss()

    def forward(self, x, y):
        return self.CELoss(x['raw'], y) + self.NLL(x['log_prob'], torch.argmax(y, dim=1))


class Stage2Loss(nn.Module):
    def __init__(self, contrastive_loss='NormSoftmax', temperature=0.05):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        if contrastive_loss == 'NormSoftmax':
            self.contrastive_loss = NormSoftmaxLoss(temperature=temperature)
        elif contrastive_loss == 'MMS':
            self.contrastive_loss = MMS_Loss()
        else:
            raise NotImplementedError()

    def forward(self, x):
        contrastive_loss = self.contrastive_loss(sim_matrix(x['data_embed'], x['text_embed']))
        return contrastive_loss
