import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

def sigmoid_focal_loss(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)
    return -(t == class_range).float() * term1 * alpha - (t != class_range).float() * term2 * (1 - alpha)

def binary_sigmoid_focal_loss(logits, targets, gamma, alpha):
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    f_loss = alpha * (1-pt)**gamma * bce_loss
    return f_loss

class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        if hasattr(gamma, '__getitem__'):
            self.gamma = gamma[0]
        else:
            self.gamma = gamma
        if hasattr(alpha, '__getitem__'):
            self.alpha = alpha[0]
        else:
            self.alpha = alpha

    def forward(self, logits, targets, **kwargs):
        # args are ignored
        device = logits.device
        loss = sigmoid_focal_loss(logits, targets.int(), self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr

class BinarySigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(BinarySigmoidFocalLoss, self).__init__()
        if hasattr(gamma, '__getitem__'):
            self.gamma = gamma[0]
        else:
            self.gamma = gamma
        if hasattr(alpha, '__getitem__'):
            self.alpha = alpha[0]
        else:
            self.alpha = alpha

    def forward(self, logits, targets, **kwargs):
        loss = binary_sigmoid_focal_loss(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr

