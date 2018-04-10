#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
from skimage import measure
from skimage.morphology import binary_dilation
from skimage import util


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = 'input/stage1_train/'
TEST_PATH = 'input/stage1_test/'
BATCH_SIZE = 32

def make_contour(img):
    contours = measure.find_contours(img, 0.8,'high','high')[0]
    cont_img = np.zeros(img.shape)
    contours = np.floor(contours).astype(int)
    for i in contours:
        cont_img[i[0],i[1]] = 1
    return cont_img



# Test informs is a list contain the id, height and width of an image
# It could be used for resize back the prediction
def make_test_data_frame(test_path):
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    test_ids = next(os.walk(test_path))[1]
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    test_informs = []
    print('Getting and resizing %d test images and masks ... '%len(test_ids))
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = test_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        test_info = (id_, img.shape[0],img.shape[1])        
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img/255
        test_informs.append(test_info)
    return X_test, test_informs

# create a numpy array of image and mask file
def make_train_data_frame(train_path):
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    train_ids = next(os.walk(train_path))[1]
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    print('Getting and resizing %d train images and masks ... '%len(train_ids))
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = train_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img/255
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask/255
    return X_train, Y_train


def make_train_data_frame_with_contour(train_path):
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    train_ids = next(os.walk(train_path))[1]
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    
    print('Getting and resizing %d train images and masks ... '%len(train_ids))
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = train_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img/255
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            contours_ = make_contour(mask_)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                          preserve_range=True), axis=-1)
            contours_ = np.expand_dims(resize(contours_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                              preserve_range=True), axis=-1)
            if (np.any(np.logical_and(mask,contours_)==True)):
                mask = np.maximum(mask, mask_)
                mask = np.minimum(mask, util.invert(contours_))
            else:
                mask = np.maximum(mask, mask_)
        Y_train[n] = mask/255
    return X_train, Y_train

def make_train_data_contour(train_path):
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    train_ids = next(os.walk(train_path))[1]
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    C_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    
    print('Getting and resizing %d train images and masks ... '%len(train_ids))
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = train_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img/255
        contours = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            contours_ = make_contour(mask_)
            contours_ = np.expand_dims(resize(contours_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                              preserve_range=True), axis=-1)
            contours = np.maximum(contours, contours_)
        C_train[n] = contours/255
    return X_train, C_train

# generate data augmentation and batch for iterations
def data_generator(X, Y, batch_size):
    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(X, seed=7)
    mask_datagen.fit(Y, seed=7)
    image_generator = image_datagen.flow(X, batch_size=batch_size, seed=7)
    mask_generator = mask_datagen.flow(Y, batch_size=batch_size, seed=7)
    feed_zip = zip(image_generator, mask_generator)
    return feed_zip

if __name__ == "__main__":
#    X_train, Y_train = make_train_data_frame(TRAIN_PATH)
    X_train, C_train = make_train_data_contour(TRAIN_PATH)
    np.save('X_train',X_train)
    np.save('C_train',C_train)
#    X_test,_ = make_test_data_frame(TEST_PATH)
#    np.save('X_test',X_test)
