#!/usr/bin/env python
from __future__ import division
import tensorflow as tf
import glob
import numpy as np
import os
import cv2
from PIL import Image, ImageSequence
from tf_record import create_tf_record
import matplotlib.pyplot as plt

def adjust_labels(np_masks):
    adjusted_masks = []
    for i in range(np.shape(np_masks)[0]):
        mask = np.squeeze(np_masks[i,:,:])
        adjusted_mask = cv2.Canny(mask,0.01,0.01)
        idx = np.where(mask == 0)
        adjusted_mask[idx] = 255
        kernel = np.ones((3,3),np.uint8)
        adjusted_mask = cv2.dilate(adjusted_mask,kernel,iterations = 1)
        adjusted_mask = np.invert(adjusted_mask)
        adjusted_masks.append(adjusted_mask)
        # f, (ax1,ax2,ax3) = plt.subplots(1,3)
        # ax1.imshow(np.squeeze(np_volume[i,:,:]), cmap='gray')
        # ax1.set_title('image')
        # ax2.imshow(mask)
        # ax2.set_title('labels')
        # ax3.imshow(adjusted_mask, cmap='gray')
        # ax3.set_title('binary mask')
        # plt.show()

    adjusted_masks = np.array(adjusted_masks)
    return adjusted_masks

    # hf = h5py.File(os.path.join(data_filepath,'updated_Berson.h5'), 'w')
    # hf.create_dataset('volume', data=volume)
    # hf.create_dataset('masks', data=masks)
    # hf.create_dataset('binary_masks', data=adjusted_masks)
    # hf.close()


def build_tfrecords(data_filepath, tfrecords_filepath):
    """
    Reads data from the data file path and produces tfrecords for training, validation 
    and test data
    Inputs: data_filepath - filepath to hdf5 for volume containing images and masks 
            tfrecords_filepath - directory to store produced tf records
    Ouputs: None
    """

    volume = []
    volume_tif = Image.open(os.path.join(data_filepath,'ISBI_2013_data/train/train-input.tif'))
    for img in ImageSequence.Iterator(volume_tif):
        volume.append(np.array(img))
    np_volume = np.array(volume)

    masks = []
    masks_tif = Image.open(os.path.join(data_filepath,'ISBI_2013_data/train/train-labels.tif'))
    for m in ImageSequence.Iterator(masks_tif):
        masks.append(np.array(m))
    np_masks = np.array(masks)

    train_volume = np_volume[:80]
    train_masks = np_masks[:80]
    create_tf_record(os.path.join(main_tfrecords_dir,'MIT_train.tfrecords'), train_volume, train_masks)

    val_volume = np_volume[80:]
    val_masks = np_masks[80:]
    create_tf_record(os.path.join(main_tfrecords_dir,'MIT_val.tfrecords'), val_volume, val_masks)

if __name__ == '__main__':
    main_data_dir = '/media/data_cifs/andreas/connectomics/ISBI_2013_data/'
    main_tfrecords_dir = '/media/data_cifs/andreas/connectomics/tfrecords/'

    build_tfrecords(os.path.join(main_data_dir,'train'),main_tfrecords_dir)
    