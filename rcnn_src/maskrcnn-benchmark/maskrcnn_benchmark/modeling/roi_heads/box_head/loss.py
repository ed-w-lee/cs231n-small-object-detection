# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import json
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.layers import (
    SigmoidFocalLoss,
    SigmoidReducedFocalLoss,
    SigmoidAreaReducedFocalLoss,
)
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self, 
        proposal_matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_loss,
        cls_agnostic_bbox_reg=False,
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.cls_loss = cls_loss

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        areas = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            areas_per_image = matched_targets.area()

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            areas.append(areas_per_image)

        return labels, regression_targets, areas

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets, areas = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, areas_per_image, proposals_per_image in zip(
            labels, regression_targets, areas, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
            proposals_per_image.add_field("areas", areas_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )
        areas = cat([proposal.get_field("areas") for proposal in proposals], dim=0)

        # classification_loss = F.cross_entropy(class_logits, labels)
        classification_loss = self.cls_loss['fn'](
            class_logits, labels, areas=areas
        )
        if self.cls_loss['avg']:
            classification_loss /= labels.numel()

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    cls_loss_fn_type = cfg.MODEL.ROI_HEADS.CLASSIFICATION_LOSS_FN
    cls_loss = {}
    if cls_loss_fn_type == "CE":
        cls_loss['fn'] = F.cross_entropy
        cls_loss['avg'] = False
    elif cls_loss_fn_type == "Focal":
        cls_loss['fn'] = SigmoidFocalLoss(
            cfg.MODEL.ROI_HEADS.FOCAL_LOSS_GAMMA,
            cfg.MODEL.ROI_HEADS.FOCAL_LOSS_ALPHA,
        )
        cls_loss['avg'] = True
    elif cls_loss_fn_type == "ReducedFocal":
        cls_loss['fn'] = SigmoidReducedFocalLoss(
            cfg.MODEL.ROI_HEADS.FOCAL_LOSS_GAMMA,
            cfg.MODEL.ROI_HEADS.FOCAL_LOSS_ALPHA,
            cfg.MODEL.ROI_HEADS.REDUCED_FOCAL_LOSS_CUTOFF,
        )
        cls_loss['avg'] = True
    elif cls_loss_fn_type == "Class":
        # me being a lazy fuck
        # counts_dict = {6: 895135, 49: 1414550, 10: 54917, 54: 22780, 52: 5242, 11: 26608, 48: 5792, 7: 30454, 14: 17734, 3: 3190, 53: 7473, 51: 5739, 43: 3053, 5: 10108, 12: 16454, 17: 790, 56: 7281, 60: 464, 9: 15035, 13: 3403, 58: 5736, 57: 11459, 36: 836, 39: 1670, 20: 7744, 45: 1411, 19: 5213, 15: 4169, 8: 4671, 44: 4012, 25: 5254, 24: 3585, 29: 3201, 40: 5770, 34: 880, 26: 3107, 47: 2846, 59: 1609, 16: 582, 46: 248, 42: 464, 55: 594, 32: 1740, 27: 1089, 30: 1037, 28: 934, 50: 1285, 1: 322, 2: 1909, 4: 359, 38: 269, 37: 279, 35: 897, 23: 475, 22: 521, 31: 2504, 21: 442, 18: 70, 41: 1230, 33: 439}
        raise ValueError("deprecated class loss")
    elif cls_loss_fn_type == "AreaFocal":
        cls_loss['fn'] = SigmoidAreaReducedFocalLoss(
            cfg.MODEL.ROI_HEADS.FOCAL_LOSS_GAMMA,
            cfg.MODEL.ROI_HEADS.FOCAL_LOSS_ALPHA,
            cfg.MODEL.ROI_HEADS.AREA_LOSS_BETA,
            cfg.MODEL.ROI_HEADS.REDUCED_FOCAL_LOSS_CUTOFF,
            cfg.MODEL.ROI_HEADS.AREA_LOSS_THRESHOLD,
        )
        cls_loss['avg'] = True
    else:
        raise ValueError("invalid classification loss type: {}".format(cls_loss_fn_type))

    loss_evaluator = FastRCNNLossComputation(
        matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_loss,
        cls_agnostic_bbox_reg,
    )

    return loss_evaluator
