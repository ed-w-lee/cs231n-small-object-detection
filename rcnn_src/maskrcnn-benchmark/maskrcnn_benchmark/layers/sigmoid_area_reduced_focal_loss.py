import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

def _get_area_weights(areas, beta, area_thresh):
    areas[areas <= area_thresh] = area_thresh - 1
    area_weights = (areas <= area_thresh).float() * 1. \
        + (areas > area_thresh).float() * torch.pow(area_thresh / areas, beta)
    return area_weights

def sigmoid_area_reduced_focal_loss(logits, targets, areas, gamma, alpha, beta, cutoff, area_thresh):
    num_classes = logits.shape[1]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    term1coef = (p < cutoff).float()*1 \
            + (p >= cutoff).float()*((1.-p)/cutoff)**gamma
    term2coef = (p > 1.-cutoff).float()*1 \
            + (p <= 1.-cutoff).float()*(p/cutoff)**gamma
    term1 = term1coef * torch.log(p)
    term2 = term2coef * torch.log(1 - p)
    rf_loss = -(t == class_range).float() * term1 * alpha - (t != class_range).float() * term2 * (1 - alpha)

    area_weights = _get_area_weights(areas, beta, area_thresh)
    return area_weights * rf_loss

def binary_sigmoid_area_reduced_focal_loss(logits, targets, areas, gamma, alpha, beta, cutoff, area_thresh):
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    rf_loss = (pt < cutoff).float() * alpha * bce_loss \
            + (pt >= cutoff).float() * alpha * ((1-pt)/cutoff)**gamma * bce_loss
    area_weights = _get_area_weights(areas, beta, area_thresh)
    return area_weights * rf_loss

def _getconst(val):
    return val[0] if hasattr(val, '__getitem__') else val

class SigmoidAreaReducedFocalLoss(nn.Module):
    def __init__(self, gamma, alpha, beta, cutoff, area_thresh):
        super(SigmoidAreaReducedFocalLoss, self).__init__()
        self.gamma = _getconst(gamma)
        self.alpha = _getconst(alpha)
        self.beta = _getconst(beta)
        self.cutoff = _getconst(cutoff)
        self.area_thresh = _getconst(area_thresh)

    def forward(self, logits, targets, areas, **kwargs):
        # args are ignored
        loss = sigmoid_area_reduced_focal_loss(logits, targets.int(), areas, 
                self.gamma, self.alpha, self.beta, self.cutoff, self.area_thresh)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ", beta=" + str(self.beta)
        tmpstr += ", cutoff=" + str(self.cutoff)
        tmpstr += ", area_thresh=" + str(self.area_thresh)
        tmpstr += ")"
        return tmpstr

class BinarySigmoidAreaReducedFocalLoss(nn.Module):
    def __init__(self, gamma, alpha, beta, cutoff, area_thresh):
        super(BinarySigmoidAreaReducedFocalLoss, self).__init__()
        self.gamma = _getconst(gamma)
        self.alpha = _getconst(alpha)
        self.beta = _getconst(beta)
        self.cutoff = _getconst(cutoff)
        self.area_thresh = _getconst(area_thresh)

    def forward(self, logits, targets, areas, **kwargs):
        # args are ignored
        loss = binary_sigmoid_area_reduced_focal_loss(logits, targets, areas, 
                self.gamma, self.alpha, self.beta, self.cutoff, self.area_thresh)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ", beta=" + str(self.beta)
        tmpstr += ", cutoff=" + str(self.cutoff)
        tmpstr += ", area_thresh=" + str(self.area_thresh)
        tmpstr += ")"
        return tmpstr

