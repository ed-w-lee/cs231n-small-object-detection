import argparse
import numpy as np
import PIL

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_folder', help='path to images')
    parser.add_argument('annotations', help='path to annotations')
    parser.add_argument('boxes', help='path to bounding boxes')
    return parser

def draw_image():

if __name__=='__main__':
    args = get_parser().parse_args()
