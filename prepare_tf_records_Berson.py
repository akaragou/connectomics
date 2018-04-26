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

def adjust_files(data_filepath):
    f1 = h5py.File(os.path.join(data_filepath,'training_set.h5'), 'r')
    f2 = h5py.File(os.path.join(data_filepath,'seg_1s.h5'), 'r')
    volume =  f1['dataset1'][:].astype('uint8')
    masks = f2['seg1s'][:].astype('uint8')

    adjusted_masks = []
    for i in range(np.shape(masks)[0]):
        mask = np.squeeze(masks[i,:,:])
        adjusted_mask = cv2.Canny(mask,0.01,0.01)
        idx = np.where(mask == 0)
        adjusted_mask[idx] = 255
        kernel = np.ones((3,3),np.uint8)
        adjusted_mask = cv2.dilate(adjusted_mask,kernel,iterations = 1)
        adjusted_mask = np.invert(adjusted_mask)
        adjusted_masks.append(adjusted_mask)
        # plt.imshow(mask)
        # plt.pause(0.5)
        # plt.imshow(adjusted_mask, cmap='gray')
        # plt.pause(0.5)

    adjusted_masks = np.array(adjusted_masks)

    hf = h5py.File(os.path.join(data_filepath,'updated_Berson.h5'), 'w')
    hf.create_dataset('volume', data=volume)
    hf.create_dataset('masks', data=masks)
    hf.create_dataset('binary_masks', data=adjusted_masks)
    hf.close()

def build_tfrecords(data_filepath, tfrecords_filepath):
    """
    Reads data from the data file path and produces tfrecords for training, validation 
    and test data
    Inputs: data_filepath - filepath to hdf5 for volume containing images and masks 
            tfrecords_filepath - directory to store produced tf records
    Ouputs: None
    """
    f = h5py.File(os.path.join(data_filepath,'updated_Berson.h5'), 'r')

    volume =  f['volume'][:].astype('uint8')
    masks =  f['binary_masks'][:].astype('uint8')
    
    binary_masks =  f['binary_masks'][:].astype('uint8')
    print np.shape(masks)
    train_volume = volume[:370]
    train_masks = masks[:370]
    create_tf_record(os.path.join(main_tfrecords_dir,'Berson_train.tfrecords'), train_volume, 
                            train_masks, is_img_resize = False, is_elastic_transform = False)

    val_volume = volume[370:374]
    val_masks = masks[370:374]
    create_tf_record(os.path.join(main_tfrecords_dir,'Berson_val.tfrecords'), val_volume, 
                             val_masks, is_img_resize = False, is_elastic_transform = False)

    test_volume = volume[374:]
    test_masks = masks[374:]
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
    # adjust_files(main_data_dir)