#!/usr/bin/env python
from __future__ import division
import os
import glob
import numpy as np
import tensorflow as tf
from scipy import misc
from tqdm import tqdm
from scipy import misc
import time
import random
import math
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2

# means for centering image data
ISBI_2012_MEAN = 126
BERSON_MEAN = 152

def draw_grid(im, grid_size):
    """ Draws grid lines on input images
    Inputs: im - input image
            grid_size - number of vertical/horizontal grid lines
    Output: im - image with grid lines
    """
    shape = im.shape
    for i in range(0, shape[1], grid_size):
        cv2.line(im, (i, 0), (i, shape[0]), color=(255,0,0))
    for j in range(0, shape[0], grid_size):
        cv2.line(im, (0, j), (shape[1], j), color=(255,0,0))

    return im

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

def preprocessing_ISBI_with_mask(img, masks):
    """ centers input image and scales mask for ISBI data
    Inputs: img - input batch of images
            masks - input batch of masks
    Outputs: normalized_img - center image batch
             normalized_mask - scaled mask batch
    """
    normalized_img = (img - ISBI_2012_MEAN)
    masks = tf.squeeze(masks)
    normalized_mask = tf.to_int64(masks/255.0)
    return normalized_img, normalized_mask

def preprocessing_Berson_with_mask(img, masks):
    """ centers input image and scales mask for Berson data
    Inputs: img - input batch of images
            masks - input batch of masks
    Outputs: normalized_img - center image batch
             normalized_mask - scaled mask batch
    """
    normalized_img = (img - BERSON_MEAN)
    masks = tf.squeeze(masks)
    normalized_mask = tf.to_int64(masks/255.0)
    return normalized_img, normalized_mask


def elastic_transform(image, alpha, sigma): 
    """ Elastic deformation of input image 
    Modified from: https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
    Inputs: image - input image
            alpha - scaling factor
            sigma - standard deviation of gaussian filter
    Outputs: distorted_image - image that has elastic deformation appliead to it
    """
    shape = image.shape
    dx = gaussian_filter((np.random.rand(shape[0],shape[1],shape[2]) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(shape[0],shape[1],shape[2]) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distorted_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distorted_image.reshape(shape)

def create_tf_record(tfrecords_filename, images, masks, model_img_dims=[256,256], is_img_resize = False, is_elastic_transform = False):
    """
    Creates tf records by writing image data to binary files (allows for faster 
    reading of data with threads). Meta data for the tf records is stored as well.
    Inputs: tfrecords_filename - Directory to store tfrecords
            images - stack of images
            masks - stack of masks
            model_img_dims - dimensions of the tensors that the model will receive
            is_img_resize - boolean to resize images to a specific dimension (tf_resize may be broken)
            is_elastic_transform - elastic deformation 
    Output: None
    """
    if not is_elastic_transform:

        writer = tf.python_io.TFRecordWriter(tfrecords_filename)
        for i in tqdm(range(np.shape(images)[0])):

            image = images[i]
            mask = masks[i]

            if is_img_resize:  
                image = misc.imresize(image, (256, 256))
                mask = misc.imresize(mask, (256, 256))

            img_raw = image.tostring()
            m_raw = mask.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
        
                    'image_raw': _bytes_feature(img_raw),
                    'mask_raw':_bytes_feature(m_raw),

                   }))

            writer.write(example.SerializeToString())

        writer.close()

    else:

        writer = tf.python_io.TFRecordWriter(tfrecords_filename)
        for _ in tqdm(range(100)):

            for i in range(np.shape(images)[0]):

                image = images[i]
                mask = masks[i]

                if is_img_resize:
                    image = misc.imresize(image, (256, 256))
                    mask = misc.imresize(mask, (256, 256))

                image = draw_grid(image, 50)
                image_mask = np.concatenate((image[...,None], mask[...,None]), axis=2)
                a = np.random.uniform(0.06,0.08)
                image_mask = elastic_transform(image_mask,  model_img_dims[0] * 3, model_img_dims[0] * a)      
                image = image_mask[...,0]
                mask = image_mask[...,1]


                img_raw = image.tostring()
                m_raw = mask.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
            
                        'image_raw': _bytes_feature(img_raw),
                        'mask_raw':_bytes_feature(m_raw),

                       }))

                writer.write(example.SerializeToString())

        writer.close()


    print '-' * 90
    print 'Generated tfrecord at %s' % tfrecords_filename
    print '-' * 90


def read_and_decode(filename_queue=None, img_dims=[384,384], size_of_batch=4,\
                     augmentations_dic=None, num_of_threads=2, shuffle=True):
    """
    Reads in tf records and decodes the features of the image 
    Input: filename_queue - a node in a TensorFlow Graph used for asynchronous computations
           img_dims - dimensions of the tensor image, example: [384,384] 
           size_of_batch - size of the batch that will be fed into the model, example: 4
           augmentations_dic - Dictionary of augmentations that an image can have for training and validation. Augmentations
           are chosen in the config
           num_threads - number of threads that execute a training op that dequeues mini-batches from the queue 
           shuffle - boolean wheter to randomly shuffle images while feeding them to the graph 
    Outputs: image - image after augmentations 
             mask - mask after augmentations 
    """
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
    
      features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)
        
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    mask = tf.decode_raw(features['mask_raw'], tf.uint8)

    image = tf.reshape(image, img_dims)
    mask = tf.reshape(mask, img_dims)

    image = tf.to_float(image)
    mask = tf.to_float(mask)

    image = tf.expand_dims(image, -1)
    mask = tf.expand_dims(mask,-1) 

    image_mask = tf.concat([image, mask], axis=-1)

    if augmentations_dic['rand_flip_left_right']:
        image_mask = tf.image.random_flip_left_right(image_mask)
        image = image_mask[...,0]
        mask = image_mask[...,1]
        image = tf.expand_dims(image, -1)
        mask = tf.expand_dims(mask,-1)

    image_mask = tf.concat([image, mask], axis=-1)
    if augmentations_dic['rand_flip_top_bottom']:     
        image_mask = tf.image.random_flip_up_down(image_mask)
        image = image_mask[...,0]
        mask = image_mask[...,1]
        image = tf.expand_dims(image, -1)
        mask = tf.expand_dims(mask,-1) 


    if augmentations_dic['rand_rotate']:
        random_angle = random.choice([0,90,180,270])
        image = tf.contrib.image.rotate(image, math.radians(random_angle))
        mask = tf.contrib.image.rotate(mask, math.radians(random_angle))
        

    if shuffle:
        image, mask = tf.train.shuffle_batch([image, mask],
                                           batch_size=size_of_batch,
                                           capacity=10000 + 3 * size_of_batch,
                                           min_after_dequeue=1000,
                                           num_threads=num_of_threads)
    else:
        image, mask = tf.train.batch([image, mask],
                                   batch_size=size_of_batch,
                                   capacity=10000,
                                   allow_smaller_final_batch=True,
                                   num_threads=num_of_threads)
      
    return image, mask
