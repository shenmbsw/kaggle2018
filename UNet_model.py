#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

def get_variable(name,shape):
    return tf.get_variable(name, shape, initializer = tf.contrib.layers.xavier_initializer())

def UNet(X):
    ### Unit 1 ###
    with tf.name_scope('Unit1'):
        W1_1 =   get_variable("W1_1", [3,3,3,16])
        Z1 = tf.nn.conv2d(X,W1_1, strides = [1,1,1,1], padding = 'SAME')
        A1 = tf.nn.elu(Z1)
        W1_2 =   get_variable("W1_2", [3,3,16,16])
        Z2 = tf.nn.conv2d(A1,W1_2, strides = [1,1,1,1], padding = 'SAME')
        A2 = tf.nn.elu(Z2) 
        P1 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 2 ###
    with tf.name_scope('Unit2'):
        W2_1 =   get_variable("W2_1", [3,3,16,32])
        Z3 = tf.nn.conv2d(P1,W2_1, strides = [1,1,1,1], padding = 'SAME')
        A3 = tf.nn.elu(Z3)
        W2_2 =   get_variable("W2_2", [3,3,32,32])
        Z4 = tf.nn.conv2d(A3,W2_2, strides = [1,1,1,1], padding = 'SAME')
        A4 = tf.nn.elu(Z4) 
        P2 = tf.nn.max_pool(A4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 3 ###
    with tf.name_scope('Unit3'):
        W3_1 =   get_variable("W3_1", [3,3,32,64])
        Z5 = tf.nn.conv2d(P2,W3_1, strides = [1,1,1,1], padding = 'SAME')
        A5 = tf.nn.elu(Z5)
        W3_2 =   get_variable("W3_2", [3,3,64,64])
        Z6 = tf.nn.conv2d(A5,W3_2, strides = [1,1,1,1], padding = 'SAME')
        A6 = tf.nn.elu(Z6) 
        P3 = tf.nn.max_pool(A6, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 4 ###
    with tf.name_scope('Unit4'):
        W4_1 =   get_variable("W4_1", [3,3,64,128])
        Z7 = tf.nn.conv2d(P3,W4_1, strides = [1,1,1,1], padding = 'SAME')
        A7 = tf.nn.elu(Z7)
        W4_2 =   get_variable("W4_2", [3,3,128,128])
        Z8 = tf.nn.conv2d(A7,W4_2, strides = [1,1,1,1], padding = 'SAME')
        A8 = tf.nn.elu(Z8) 
        P4 = tf.nn.max_pool(A8, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 5 ###
    with tf.name_scope('Unit5'):
        W5_1 =   get_variable("W5_1", [3,3,128,256])
        Z9 = tf.nn.conv2d(P4,W5_1, strides = [1,1,1,1], padding = 'SAME')
        A9 = tf.nn.elu(Z9)
        W5_2 =   get_variable("W5_2", [3,3,256,256])
        Z10 = tf.nn.conv2d(A9,W5_2, strides = [1,1,1,1], padding = 'SAME')
        A10 = tf.nn.elu(Z10)
    ### Unit 6 ###
    with tf.name_scope('Unit6'):
        U1 = tf.image.resize_images(A10, (16, 16))
        U1 = tf.concat([U1, A8],3)
        W6_1 =   get_variable("W6_1", [3,3,384,128])
        Z11 = tf.nn.conv2d(U1,W6_1, strides = [1,1,1,1], padding = 'SAME')
        A11 = tf.nn.elu(Z11)
        W6_2 =   get_variable("W6_2", [3,3,128,128])
        Z12 = tf.nn.conv2d(A11, W6_2, strides = [1,1,1,1], padding = 'SAME')
        A12 = tf.nn.elu(Z12)
    ### Unit 7 ###
    with tf.name_scope('Unit7'):
        U2 = tf.image.resize_images(A12, [32, 32])
        U2 = tf.concat([U2, A6],3)
        W7_1 =   get_variable("W7_1", [3,3,192,64])
        Z13 = tf.nn.conv2d(U2,W7_1, strides = [1,1,1,1], padding = 'SAME')
        A13 = tf.nn.elu(Z13)
        W7_2 =   get_variable("W7_2", [3,3,64,64])
        Z14 = tf.nn.conv2d(A13,W7_2, strides = [1,1,1,1], padding = 'SAME')
        A14 = tf.nn.elu(Z14)
    ### Unit 8 ###
    with tf.name_scope('Unit8'):
        U3 = tf.image.resize_images(A14, [64, 64])
        U3 = tf.concat([U3, A4],3)
        W8_1 =   get_variable("W8_1", [3,3,96,32])
        Z15 = tf.nn.conv2d(U3,W8_1, strides = [1,1,1,1], padding = 'SAME')
        A15 = tf.nn.elu(Z15)
        W8_2 =   get_variable("W8_2", [3,3,32,32])
        Z16 = tf.nn.conv2d(A15,W8_2, strides = [1,1,1,1], padding = 'SAME')
        A16 = tf.nn.elu(Z16)
    ### Unit 9 ###
    with tf.name_scope('Unit9'):
        U4 = tf.image.resize_images(A16, [128, 128])
        U4 = tf.concat([U4, A2],3)
        W9_1 =   get_variable("W9_1", [3,3,48,16])
        Z17 = tf.nn.conv2d(U4,W9_1, strides = [1,1,1,1], padding = 'SAME')
        A17 = tf.nn.elu(Z17)
        W9_2 =   get_variable("W9_2", [3,3,16,16])
        Z18 = tf.nn.conv2d(A17,W9_2, strides = [1,1,1,1], padding = 'SAME')
        A18 = tf.nn.elu(Z18)
    ### Unit 10 ###
    with tf.name_scope('out_put'):
        W10 =    get_variable("W10", [1,1,16,1])
        Z19 = tf.nn.conv2d(A18,W10, strides = [1,1,1,1], padding = 'SAME')
        A19 = tf.nn.sigmoid(Z19)
        Y_pred = A19
    return Y_pred

def dice_loss(y_pred, y_true):
    intercept = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true))
    union = tf.add(tf.reduce_sum(tf.square(y_pred)),tf.reduce_sum(tf.square(y_true)))
    return 1 - tf.divide(intercept,union)

def loss_function(y_pred, y_true):
    cost = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true,y_pred)) + dice_loss(y_pred, y_true)
    return cost

def mean_iou(y_pred,y_true):
    y_pred_ = tf.to_int64(y_pred > 0.3)
    y_true_ = tf.to_int64(y_true > 0.3)
    score, up_opt = tf.metrics.mean_iou(y_true_, y_pred_, 2)
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score