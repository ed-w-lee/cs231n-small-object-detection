import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

def _get_area_weights(areas, beta, area_thresh):
    areas[areas == 0] = -1 # no nans!
    area_offs = areas - area_thresh
    area_offs[area_offs == 0] = -1 # no nans!
    area_weights = (area_offs <= 0).float() * 1. \
            + (area_offs > 0).float() * torch.pow(area_thresh / areas, beta)
    return area_weights

def area_loss(logits, targets, areas, beta, area_thresh):
    area_weights = _get_area_weights(areas, beta, area_thresh).unsqueeze(1)
    return F.cross_entropy(logits, targets, weight=area_weights, reduction='none')

def binary_area_loss(logits, targets, areas, beta, area_thresh):
    area_weights = _get_area_weights(areas, beta, area_thresh)
    return F.binary_cross_entropy_with_logits(logits, targets, weight=area_weights, reduction='none')

def _getconst(val):
    return val[0] if hasattr(val, '__getitem__') else val

class AreaLoss(nn.Module):
    def __init__(self, beta, area_thresh):
        super(SigmoidAreaReducedFocalLoss, self).__init__()
        self.beta = _getconst(beta)
        self.area_thresh = _getconst(area_thresh)

    def forward(self, logits, targets, areas, **kwargs):
        # args are ignored
        loss = area_loss(logits, targets.int(), areas, 
                self.beta, self.area_thresh)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += ", beta=" + str(self.beta)
        tmpstr += ", area_thresh=" + str(self.area_thresh)
        tmpstr += ")"
        return tmpstr

class BinaryAreaLoss(nn.Module):
    def __init__(self, beta, area_thresh):
        super(BinaryAreaLoss, self).__init__()
        self.beta = _getconst(beta)
        self.area_thresh = _getconst(area_thresh)

    def forward(self, logits, targets, areas, **kwargs):
        # args are ignored
        loss = binary_area_loss(logits, targets, areas, 
                self.beta, self.area_thresh)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += ", beta=" + str(self.beta)
        tmpstr += ", area_thresh=" + str(self.area_thresh)
        tmpstr += ")"
        return tmpstr

