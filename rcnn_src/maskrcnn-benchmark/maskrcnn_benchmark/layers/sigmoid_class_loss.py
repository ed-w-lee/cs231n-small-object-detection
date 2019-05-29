import torch
from torch import nn

def sigmoid_class_loss(logits, targets, n_targets, gamma, beta):
    num_classes = logits.shape[1]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes+1, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    term1coef = (1.-p)**gamma * (1-beta)/(1-beta**n_targets)
    term2coef = p**gamma * (1-beta)/(1-beta**n_targets)
    term1 = term1coef * torch.log(p)
    term2 = term2coef * torch.log(1 - p)
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
        self.counts = torch.zeros(max(counts_dict) + 1)
        keys = torch.tensor(list(counts_dict.keys()))
        values = torch.tensor(list(counts_dict.values()))
        self.counts[keys] = values.float()

    def forward(self, logits, targets, **kwargs):
        # args are ignored
        targets = targets.int()
        loss = sigmoid_class_loss(logits, targets, self.counts[targets.long()], self.gamma, self.beta)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", beta=" + str(self.beta)
        tmpstr += ")"
        return tmpstr
