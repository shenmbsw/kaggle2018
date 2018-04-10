#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from UNet_model import UNet
from matplotlib import pyplot as plt
from train_model import build_graph
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# this function is for produce mask prediction for no label input
def apply_model_to_test(model_ckpt, X_test):
    with tf.Graph().as_default():
        with tf.device("/cpu:0"):
            with tf.Session() as sess:
                x_ = tf.placeholder(tf.float32, shape=(None,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
                y_pred = UNet(x_)
                saver = tf.train.Saver()
                saver.restore(sess, model_ckpt)
                sess.run(tf.local_variables_initializer())
                print('start apply model to test data')
                y_pred = sess.run(y_pred, feed_dict ={x_: X_test})
    return y_pred

def validate_model(model_dict, X_train, Y_train, model_dir):
    with model_dict['graph'].as_default():
        saver = tf.train.Saver(max_to_keep=0)
        with tf.Session() as sess:
            saver.restore(sess, model_dir)
            Iou_result = model_dict['Iou']
            Loss_result = model_dict['Loss']
            y_pred = model_dict['y_pred']
            sess.run(tf.local_variables_initializer())
            train_feed_dict = dict(zip(model_dict['inputs'], (X_train, Y_train)))
            Y_pred, tr_iou,tr_loss = sess.run([y_pred,Iou_result, Loss_result], feed_dict = train_feed_dict)
    return Y_pred, tr_iou,tr_loss

if __name__ == "__main__":
    model_dict = build_graph()
    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')
    model="./Model/UNet_train5.ckpt"
    xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=7)
    Y_pred, val_iou, val_loss = validate_model(model_dict, xval, yval, model)
    num = 9
    valid_idx = num #random.randint(0,9)
    test_idx = num #random.randint(0,9)

    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax1.imshow(xval[valid_idx,:,:])
    ax2 = fig.add_subplot(232)
    ax2.imshow(yval[valid_idx,:,:,0],cmap='gray')
    ax3 = fig.add_subplot(233)
    ax3.imshow(Y_pred[valid_idx,:,:,0],cmap='gray')
    print(val_iou, val_loss)
    
    X_test = np.load('X_test.npy')[0:10,:,:,:]
    Y_pred = apply_model_to_test(model, X_test )
    
    ax4 = fig.add_subplot(234)
    ax4.imshow(X_test[test_idx,:,:])
    ax5 = fig.add_subplot(235)
    ax5.imshow(Y_pred[test_idx,:,:,0],cmap='gray')
    thre = 0.1
    Y_binary_pred = (Y_pred>thre)
    ax6 = fig.add_subplot(236)
    ax6.imshow(Y_binary_pred[test_idx,:,:,0],cmap='gray')
    fig.savefig('figure%d.png'%num)

    
    
