"""
Copyright 2018 Defense Innovation Unit Experimental
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Edited by Edward Lee 2019
"""

from tqdm import tqdm
from PIL import Image
import aug_util as aug
import wv_util as wv
import numpy as np
import argparse
import logging
import random
import glob
import json
import csv
import os
import io

"""
  A script that processes xView imagery. 
  Args:
      image_folder: A folder path to the directory storing xView .tif files
        ie ("xView_data/")

      json_filepath: A file path to the GEOJSON ground truth file
        ie ("xView_gt.geojson")

      test_percent (-t): The percentage of input images to use for test set

      suffix (-s): The suffix for output TFRecord files.  Default suffix 't1' will output
        xview_train_t1.record and xview_test_t1.record

      augment (-a): A boolean value of whether or not to use augmentation

  Outputs:
    Writes two files to the current directory containing training and test data in
        TFRecord format ('xview_train_SUFFIX.record' and 'xview_test_SUFFIX.record')
"""


def shuffle_images_and_boxes_classes(im,box,cls):
    """
    Shuffles images, boxes, and classes, while keeping relative matching indices

    Args:
        im: an array of images
        box: an array of bounding box coordinates ([xmin,ymin,xmax,ymax])
        cls: an array of classes

    Output:
        Shuffle image, boxes, and classes arrays, respectively
    """
    assert len(im) == len(box)
    assert len(box) == len(cls)
    
    perm = np.random.permutation(len(im))
    out_b = {}
    out_c = {}
    
    k = 0 
    for ind in perm:
        out_b[k] = box[ind]
        out_c[k] = cls[ind]
        k = k + 1
    return im[perm], out_b, out_c

def output_img(im, c_box, c_cls, out_train_dir, fname, idx, image_id, ann_id_start, cat_list):
    img = Image.fromarray(im, 'RGB')
    out_file = "%s.jpg" % image_id
    img.save(os.path.join(out_train_dir, out_file))
    img_dict = {}
    img_dict['file_name'] = out_file
    img_dict['height'] = im.shape[0]
    img_dict['width'] = im.shape[1]
    img_dict['id'] = image_id
    annos = []
    for box,cls in zip(c_box, c_cls):
        if (box[2]-box[0] <= 2) or (box[3]-box[1] <= 2):
            continue
        if cls not in cat_list:
            continue
        box_dict = {}
        box_dict['id'] = ann_id_start
        ann_id_start += 1
        box_dict['image_id'] = image_id
        box_dict['bbox'] = [int(box[0]),int(box[1]),int(box[2]-box[0]),int(box[3]-box[1])]
        box_dict['iscrowd'] = 0
        box_dict['category_id'] = int(cls)
        box_dict['area'] = box_dict['bbox'][2] * box_dict['bbox'][3]
        annos.append(box_dict)
    img_dict['from'] = fname
    img_dict['idx'] = idx
    return img_dict, annos, ann_id_start

'''
Datasets
_multires: multiple resolutions. Currently [(500,500),(400,400),(300,300),(200,200)]
_aug: Augmented dataset
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder", help="Path to folder containing image chips (ie 'Image_Chips/' ")
    parser.add_argument("json_filepath", help="Filepath to GEOJSON coordinate file")
    parser.add_argument("out_folder", help="folder for output")
    parser.add_argument("category_filepath", help="filepath for categories")
    parser.add_argument("--num_train", type=int, default=594,
                    help="number to put into train")
    parser.add_argument("--num_val", type=int, default=126,
                    help="number to put into val")
    parser.add_argument("--num_test", type=int, default=-1,
                    help="number to put into test. if -1, will place non-train/val into test")
    parser.add_argument("-s", "--suffix", type=str, default='t1',
                    help="Output suffix. Default suffix 't1' will output 'xview_train_t1/' and 'xview_test_t1/'")
    parser.add_argument("-a","--augment", type=bool, default=False,
    				help="A boolean value whether or not to use augmentation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    #resolutions should be largest -> smallest.  We take the number of chips in the largest resolution and make
    #sure all future resolutions have less than 1.5times that number of images to prevent chip size imbalance.
    #res = [(500,500),(400,400),(300,300),(200,200)]
    random.seed(231)
    res = [(700,700)]
    overlap=80

    AUGMENT = args.augment
    SAVE_IMAGES = False
    images = {}
    boxes = {}
    train_chips = 0
    val_chips = 0
    test_chips = 0
    train_cutoff = args.num_train
    val_cutoff = args.num_train + args.num_val
    test_cutoff = val_cutoff + args.num_test if args.num_test >= 0 else -1
    categories = []
    cat_list = []
    with open(args.category_filepath, 'r') as fin:
        cats = json.load(fin)
        for k,v in cats.items():
            categories.append({'id': int(k), 'name': v})
            cat_list.append(int(k))
    train_ann = {
        "annotations": [],
        "images": [],
        "categories": categories,
    }
    val_ann = {
        "annotations": [],
        "images": [],
        "categories": categories,
    }
    test_ann = {
        "annotations": [],
        "images": [],
        "categories": categories,
    }
             

    #Parameters
    max_chips_per_res = 100000
    out_folder = args.out_folder
    out_train_dir = os.path.join(out_folder, "train_%s" % args.suffix)
    out_val_dir = os.path.join(out_folder, "val_%s" % args.suffix)
    out_test_dir = os.path.join(out_folder, "test_%s" % args.suffix)
    ann_dir = os.path.join(out_folder, 'annotations_%s' % args.suffix)
    out_train_ann = "train.json"
    out_val_ann = "val.json"
    out_test_ann = "test.json"
    if os.path.exists(out_train_dir) or os.path.exists(out_val_dir) \
        or os.path.exists(out_test_dir) or os.path.exists(ann_dir):
        logger.error("at least one output dir exists, please delete")
        quit()
    else:
        os.makedirs(out_train_dir)
        os.makedirs(out_val_dir)
        os.makedirs(out_test_dir)
        os.makedirs(ann_dir)

    coords,chips,classes = wv.get_labels(args.json_filepath)

    fnames = glob.glob(os.path.join(args.image_folder,"*.tif"))
    fnames.sort()
    random.shuffle(fnames)
    ann_id = 0
    for res_ind, it in enumerate(res):
        tot_box = 0
        logging.info("Res: %s" % str(it))
        ind_chips = 0

        for f_ind, fname in enumerate(tqdm(fnames)):
            # Needs to be "X.tif", ie ("5.tif")
            name = fname.split("/")[-1]
            arr = wv.get_image(fname)

            im,box,classes_final = wv.chip_image(arr,coords[chips==name],classes[chips==name],it,
                                                 overlap=overlap)
            
            if test_cutoff >= 0 and f_ind > test_cutoff:
                break

            for idx, image in enumerate(im):
                tot_box += len(box[idx])
                if f_ind < train_cutoff:
                    img_dict, annos, ann_id = output_img(image, box[idx], classes_final[idx], 
                        out_train_dir, fname, idx, ind_chips, ann_id, cat_list)
                    train_ann['annotations'].extend(annos)
                    train_ann['images'].append(img_dict)
                    train_chips+=1
                elif f_ind < val_cutoff:
                    out_file = "%s.jpg" % ind_chips
                    img_dict, annos, ann_id = output_img(image, box[idx], classes_final[idx], 
                        out_val_dir, fname, idx, ind_chips, ann_id, cat_list)
                    val_ann['annotations'].extend(annos)
                    val_ann['images'].append(img_dict)
                    val_chips += 1
                elif test_cutoff < 0 or f_ind < test_cutoff:
                    out_file = "%s.jpg" % ind_chips
                    img_dict, annos, ann_id = output_img(image, box[idx], classes_final[idx], 
                        out_test_dir, fname, idx, ind_chips, ann_id, cat_list)
                    test_ann['annotations'].extend(annos)
                    test_ann['images'].append(img_dict)
                    test_chips += 1
     
                ind_chips +=1
#                 if AUGMENT:
#                     for extra in range(3):
#                         center = np.array([int(image.shape[0]/2),int(image.shape[1]/2)])
#                         deg = np.random.randint(-10,10)
#                         #deg = np.random.normal()*30
#                         newimg = aug.salt_and_pepper(aug.gaussian_blur(image))
# 
#                         #.3 probability for each of shifting vs rotating vs shift(rotate(image))
#                         p = np.random.randint(0,3)
#                         if p == 0:
#                             newimg,nb = aug.shift_image(newimg,box[idx])
#                         elif p == 1:
#                             newimg,nb = aug.rotate_image_and_boxes(newimg,deg,center,box[idx])
#                         elif p == 2:
#                             newimg,nb = aug.rotate_image_and_boxes(newimg,deg,center,box[idx])
#                             newimg,nb = aug.shift_image(newimg,nb)
#                             
# 
#                         newimg = (newimg).astype(np.uint8)
# 
#                         if idx%1000 == 0 and SAVE_IMAGES:
#                             Image.fromarray(newimg).save('process/img_%s_%s_%s.png'%(name,extra,it[0]))
# 
#                         if len(nb) > 0:
#                             tf_example = tfr.to_tf_example(newimg,nb,classes_final[idx])
# 
#                             #Don't count augmented chips for chip indices
#                             if idx < split_ind:
#                                 test_writer.write(tf_example.SerializeToString())
#                                 test_chips += 1
#                             else:
#                                 train_writer.write(tf_example.SerializeToString())
#                                 train_chips+=1
#                         else:
#                             if SAVE_IMAGES:
#                                 aug.draw_bboxes(newimg,nb).save('process/img_nobox_%s_%s_%s.png'%(name,extra,it[0]))
        logging.info("Tot Box: %d" % tot_box)
        logging.info("Chips: %d" % ind_chips)

    with open(os.path.join(ann_dir, out_train_ann), 'w') as fout:
        json.dump(train_ann, fout)
    with open(os.path.join(ann_dir, out_val_ann), 'w') as fout:
        json.dump(val_ann, fout)
    with open(os.path.join(ann_dir, out_test_ann), 'w') as fout:
        json.dump(test_ann, fout)

    logging.info("saved: %d train chips" % train_chips)
    logging.info("saved: %d val chips" % val_chips)
    logging.info("saved: %d test chips" % test_chips)
