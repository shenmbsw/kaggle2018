#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage import morphology
from skimage.transform import resize
from apply_model_to_test import apply_model_to_test
from data_generator import make_test_data_frame
import pandas as pd
from skimage.filters import threshold_otsu

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# this function return the pixel code of input image.
# warning, the input image should be at its original scale.
def run_length_encode(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    run_lengths = ' '.join([str(r) for r in run_lengths])
    return run_lengths

# this function return the image array with pixel code input.
def run_length_decode(rel, H, W, fill_value=255):
    mask = np.zeros((H*W),np.uint8)
    rel  = np.array([int(s) for s in rel.split(' ')]).reshape(-1,2)
    for r in rel:
        start = r[0]
        end = start +r[1]
        mask[start:end]=fill_value
    mask = mask.reshape(H,W)
    return mask

if __name__ == "__main__":
    test_path = 'input/stage1_test/'
    X_test, test_informs = make_test_data_frame(test_path)
    test_num = len(test_informs)
    csv_file = 'submission-ss.csv'
    
    
    
 
