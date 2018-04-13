#!/usr/bin/env python
from __future__ import division
import os
import tensorflow as tf
import argparse
import numpy as np
from tensorflow.contrib import slim
from config import ConnectomicsConfig
from tf_record import read_and_decode
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image

def test_tf_records(device):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device) # use nvidia-smi to see available options '0' means first gpu
    config = ConnectomicsConfig() # loads pathology configuration defined in vgg_config

    with tf.Graph().as_default():
        # load training data
        train_filename_queue = tf.train.string_input_producer(
        [config.train_fn], num_epochs=4)
        # load validation data
        val_filename_queue = tf.train.string_input_producer(
        [config.val_fn], num_epochs=12)

        train_image, train_mask = read_and_decode(filename_queue = train_filename_queue,
                                                         img_dims = [384,384],
                                                         size_of_batch = 1,
                                                          augmentations_dic = config.train_augmentations_dic,
                                                         num_of_threads = 1,
                                                         shuffle = False)

        val_image, val_mask = read_and_decode(filename_queue = val_filename_queue,
                                                     img_dims = [384,384],
                                                     size_of_batch = 1,
                                                     augmentations_dic = config.val_augmentations_dic,
                                                     num_of_threads = 1,
                                                     shuffle = False)

        count = 0 
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            sess.run(tf.group(tf.global_variables_initializer(),
                 tf.local_variables_initializer()))
           
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:

                while not coord.should_stop():

                    np_image, np_mask = sess.run([train_image, train_mask])
            
         
                    # print np.shape(np_image)
                    # print np.shape(np_mask)
                    # import pdb; pdb.set_trace()
                    # np_image = np.squeeze(np_image)
                    # np_mask = np.squeeze(np_mask)
                    # im_merge = np.concatenate((np_image[...,None], np_mask[...,None]), axis=2)
                
                    # im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 3, im_merge.shape[1] * 0.07, im_merge.shape[1] * 0.09)
                    # im_t = im_merge_t[...,0]
                    # im_mask_t = im_merge_t[...,1]
                    # plt.imshow(np.c_[np.r_[np.squeeze(np_i), np.squeeze(np_m)], np.r_[np.squeeze(np_image), np.squeeze(np_mask)]], cmap='gray')
                    plt.imshow(np.squeeze(np_image[0]), cmap='gray')
                    plt.pause(5)
                    plt.imshow(np.squeeze(np_mask[0]))
                    plt.pause(5)
                     

                    print count
                    count += 1
                 
            except tf.errors.OutOfRangeError:
                print "done testing tfrecords."
            finally:
                coord.request_stop()
            coord.join(threads)
            sess.close()
                



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    args = parser.parse_args()
    test_tf_records(args.device) # select gpu to train model on