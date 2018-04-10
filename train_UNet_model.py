#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data_generator import data_generator
from UNet_model import UNet, mean_iou, loss_function
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
SAMPLE_NUM = 670
BATCH_SIZE = 32

# build the graph as a dictionary
def build_graph():
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):
            with tf.name_scope('input'):
                x_ = tf.placeholder(tf.float32, shape=(None,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
                y_ = tf.placeholder(tf.float32, shape=(None,IMG_HEIGHT, IMG_WIDTH, 1))
            y_pred = UNet(x_)
            with tf.name_scope('loss'):
                loss = loss_function(y_pred,y_)
        with tf.device("/cpu:0"):
            with tf.name_scope("metrics"):
                iou = mean_iou(y_pred,y_)
        model_dict = {'graph': g, 'inputs': [x_, y_],'Iou':iou,'Loss':loss, 'y_pred':y_pred}
    return model_dict

# save the graph in tensorboard.
def see_graph(model_dict):
    tf.summary.FileWriter('graph/show_graph', model_dict['graph'])

# train the model and save the trained model as ckpt file
def train_model(model_dict, X_train, Y_train, max_epoch, max_iter, restore=False, valid = True):
    # split validation set or not
    if valid:
        xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=7)
        train_zip = data_generator(xtr, ytr, BATCH_SIZE)
        val_zip = data_generator(xval, yval, BATCH_SIZE)
    else:
        train_zip = data_generator(X_train, Y_train, BATCH_SIZE)
    with model_dict['graph'].as_default():
        # define model saver
        saver = tf.train.Saver(max_to_keep=0)
        with tf.Session() as sess:
            optimizer = tf.train.MomentumOptimizer(0.003, 0.9)
            trainer = optimizer.minimize(model_dict['Loss'])
            Iou_result = model_dict['Iou']
            Loss_result = model_dict['Loss']
            # flag for restore parameter
            if(restore==False):
                sess.run(tf.variables_initializer(tf.global_variables()))
            else:
                saver.restore(sess, restore)
                # the parameter of optimizer is not in saver thus initialization is required
                momentum_initializers = [var.initializer for var in tf.global_variables() if 'Momentum' in var.name]
                sess.run(momentum_initializers)
            # training start
            for epoch_i in range(max_epoch):
                print('epoch%d start...'%epoch_i)
                sess.run(tf.local_variables_initializer())
                for iter_i, train_batch in enumerate(train_zip):
                    train_feed_dict = dict(zip(model_dict['inputs'], train_batch))
                    _,tr_iou,tr_loss = sess.run([trainer, Iou_result, Loss_result], feed_dict = train_feed_dict)
                    # stop iteration and print result for this epoch
                    if (iter_i >= max_iter):
                        print('IOU:%f, loss:%f'%(tr_iou,tr_loss))
                        break
                if (epoch_i % 5 == 0):
                    sess.run(tf.local_variables_initializer())
                    # show result on validation set
                    if valid:
                        for iter_i, val_batch in enumerate(val_zip):
                            val_feed_dict = dict(zip(model_dict['inputs'], val_batch))
                            val_iou, val_loss = sess.run([Iou_result, Loss_result], feed_dict=val_feed_dict)
                            if (iter_i >= SAMPLE_NUM//BATCH_SIZE):
                                break
                        print('epoch%d valid IOU:%f, loss:%f'%(epoch_i,val_iou,val_loss))
                    # save the trained weight
                    saver.save(sess, "Model/UNet_train%d.ckpt"%epoch_i)
            #save final result
            saver.save(sess, "Model/UNet_final.ckpt")

def validate_model(model_dict, X_train, Y_train, model_dir):
    with model_dict['graph'].as_default():
        saver = tf.train.Saver(max_to_keep=0)
        with tf.Session() as sess:
            saver.restore(sess, restore)
            Iou_result = model_dict['Iou']
            Loss_result = model_dict['Loss']
            y_pred = model_dict['y_pred']
            sess.run(tf.local_variables_initializer())
            train_feed_dict = dict(zip(model_dict['inputs'], (X_train, Y_train)))
            Y_pred, tr_iou,tr_loss = sess.run([y_pred,Iou_result, Loss_result], feed_dict = train_feed_dict)
    return Y_pred, tr_iou,tr_loss

if __name__ == "__main__":
    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')
    restore = False
    model_dict = build_graph()
    see_graph(model_dict)
    train_model(model_dict, X_train, Y_train, 20, 500, restore=False)