import torch
import torch.nn as nn


class ElasticLoss(nn.Module):
    def __init__(self, alpha=0.5, l1_ratio=0.5, reduction='mean'):
        super(ElasticLoss, self).__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.reduction = reduction
        
    def forward(self, input, target):
        diff = input - target
        loss = (1 - self.l1_ratio) * 0.5 * diff.pow(2) + \
                self.l1_ratio * torch.abs(diff)
        alpha_norm = self.alpha * (
            (1 - self.l1_ratio) * 0.5 * input.pow(2) + \
            self.l1_ratio * torch.abs(input)
        ).sum(dim=1)
        loss = loss.mean(dim=1)
        loss = loss + alpha_norm
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss