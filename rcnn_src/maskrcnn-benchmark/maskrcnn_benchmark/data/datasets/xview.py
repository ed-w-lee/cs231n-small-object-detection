# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import json
import os
from collections import OrderedDict

from maskrcnn_benchmark.structures.bounding_box import BoxList

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    return True

class XViewDataset(torchvision.datasets.coco.CocoDetection):

    # constants
    SPLITS = OrderedDict([
        ('small', [17,18,19,20,21,23,24,26,27,28,32,41,60,62,63,65,66,91]),
        ('medium', [11,12,15,25,29,33,34,35,36,37,38,42,44,47,50,53,56,59,61,71,72,73,76,84,86,93,94]),
        ('large', [13,40,45,49,51,52,54,55,57,74,77,79,83,89]),
    ])

    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(XViewDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(XViewDataset, self).__getitem__(idx)

        target = self._get_groundtruth_from_img_anno(img, anno)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, idx):
        img_id = self.id_to_img_map[idx]
        img_data = self.coco.imgs[img_id]
        return img_data

    def _get_groundtruth_from_img_anno(self, img, anno):
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        target = target.clip_to_image(remove_empty=True)
        return target

    def get_groundtruth(self, idx):
        img, anno = super(XViewDataset, self).__getitem__(idx)

        target = self._get_groundtruth_from_img_anno(img, anno)
        return target

    def map_class_id_to_class_name(self, class_id):
        json_id = self.contiguous_category_id_to_json_id[class_id]
        return self.coco.loadCats(json_id)[0]['name']

    def get_splits(self):
        splits = OrderedDict()
        for k,v in self.SPLITS.items():
            splits[k] = [self.json_category_id_to_contiguous_id[cat] for cat in v]
        return splits
