# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

def sigmoid_class_loss(logits, targets, weights, gamma, beta):
    num_classes = logits.shape[1]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes+1, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    weights = weights.to(device)
    term1 = torch.log(p) * weights[t.long()-1]
    term2 = torch.log(1 - p) * weights[t.long()-1]
    return -(t == class_range).float() * term1 - ((t != class_range) * (t >= 0)).float() * term2

class SigmoidClassLoss(nn.Module):
    def __init__(self, gamma, beta, counts_dict):
        super(SigmoidClassLoss, self).__init__()
        if hasattr(gamma, '__getitem__'):
            self.gamma = gamma[0]
        else:
            self.gamma = gamma
        if hasattr(beta, '__getitem__'):
            self.beta = beta[0]
        else:
            self.beta = beta
        counts = torch.zeros(max(counts_dict))
        keys = torch.tensor(list(counts_dict.keys()))-1
        values = torch.tensor(list(counts_dict.values()))
        counts[keys] = values.float()
        weights = (1. - self.beta) / (1. - self.beta**counts)
        self.weights = weights / weights.sum()
        print(self.weights)

    def forward(self, logits, targets, **kwargs):
        # args are ignored
        targets = targets.int()
        loss = sigmoid_class_loss(logits, targets, self.weights, self.gamma, self.beta)
        # sketch shit for dayzz
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", beta=" + str(self.beta)
        tmpstr += ")"
        return tmpstr
