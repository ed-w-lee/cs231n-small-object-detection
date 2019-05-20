import os
import glob
import argparse
import shutil
import json
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image_folder") 
    parser.add_argument("input_annotations_file")
    parser.add_argument("output_image_folder")
    parser.add_argument("output_annotations_file")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    # full_im_dir = os.path.join(args.input_image_folder, "*")
    # if not args.test:
    #     os.makedirs(args.output_image_folder)
    #     os.makedirs(os.path.dirname(args.output_annotations_file))
    # for f in tqdm(glob.glob(full_im_dir)):
    #     basename = os.path.splitext(os.path.basename(f))[0]
    #     out_dir = int(basename)//1000
    #     full_out_dir = os.path.join(args.output_image_folder, str(out_dir))
    #     if args.test:
    #         print("copy {} to {}".format(f, full_out_dir))
    #         break
    #     else:
    #         if not os.path.exists(full_out_dir):
    #             os.makedirs(full_out_dir)
    #         shutil.copy(f, full_out_dir)
    
    with open(args.input_annotations_file, 'r') as fin:
        anns = json.load(fin)
        for im in tqdm(anns['images']):
            orig = im['file_name']
            basename = os.path.splitext(os.path.basename(im['file_name']))[0]
            out_dir = int(basename)//1000
            im['file_name'] = os.path.join(str(out_dir), im['file_name'])
            if args.test:
                print("original filename: {}, new filename: {}".format(orig, im['file_name']))
                break
        if not args.test:
            with open(args.output_annotations_file, 'w') as fout:
                json.dump(anns, fout)
