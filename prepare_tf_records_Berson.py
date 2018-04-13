#!/usr/bin/env python
from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import misc
import glob
import numpy as np
import os
import h5py
import cv2
from matplotlib import pyplot as plt
from tf_record import create_tf_record

def build_tfrecords(data_filepath, tfrecords_filepath):
    """
    Reads data from the data file path and produces tfrecords for training, validation 
    and test data
    Inputs: data_filepath - filepath to hdf5 for volume containing images and masks 
            tfrecords_filepath - directory to store produced tf records
    Ouputs: None
    """
    f = h5py.File(os.path.join(data_filepath,'Berson.h5'), 'r')

    volume =  f['volume'][:].astype('uint8')
    masks =  f['masks'][:].astype('uint8')
    binary_masks =  f['binary_masks'][:].astype('uint8')

    train_volume = volume[:370]
    train_masks = binary_masks[:370]
    create_tf_record(os.path.join(main_tfrecords_dir,'Berson_train.tfrecords'), train_volume, 
                            train_masks, is_img_resize = False, is_elastic_transform = False)

    val_volume = volume[370:374]
    val_masks = binary_masks[370:374]
    create_tf_record(os.path.join(main_tfrecords_dir,'Berson_val.tfrecords'), val_volume, 
                             val_masks, is_img_resize = False, is_elastic_transform = False)

    test_volume = volume[374:]
    test_masks = binary_masks[374:]
    create_tf_record(os.path.join(main_tfrecords_dir,'Berson_test.tfrecords'), test_volume,
                             test_masks, is_img_resize = False,  is_elastic_transform = False)

    n = len(np.ravel(train_volume))
    print np.sum(train_volume)/n 

    # class weights to be used for balancing classes
    flattened_masks = np.ravel(train_masks)
    unique, counts = np.unique(train_masks, return_counts=True)
    class_dic = dict(zip(unique, counts))
    f_i = map(lambda x: len(flattened_masks)/x, 
                            class_dic.values())
    class_weights =np.array(map(lambda x: x/sum(f_i), f_i), dtype=np.float32)
    np.save('class_weights_Berson.npy', class_weights)

if __name__ == '__main__':
    main_data_dir = '/media/data_cifs/andreas/connectomics/Berson'
    main_tfrecords_dir = '/media/data_cifs/andreas/connectomics/tfrecords/'

    build_tfrecords(main_data_dir,main_tfrecords_dir)