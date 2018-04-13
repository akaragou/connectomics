#!/usr/bin/env python
from __future__ import division
import tensorflow as tf
import glob
import numpy as np
import os
from PIL import Image, ImageSequence
from tf_record import create_tf_record

def build_train_val_tfrecords(data_filepath, tfrecords_filepath):
    """
    Reads data from the data file path and produces tfrecords for training, validation 
    and test data
    Inputs: data_filepath - filepath to hdf5 for volume containing images and masks 
            tfrecords_filepath - directory to store produced tf records
    Ouputs: None
    """

    volume = []
    volume_tif = Image.open(os.path.join(data_filepath,'train-volume.tif'))
    for img in ImageSequence.Iterator(volume_tif):
        volume.append(np.array(img))
    np_volume = np.array(volume)

    masks = []
    masks_tif = Image.open(os.path.join(data_filepath,'train-labels.tif'))
    for m in ImageSequence.Iterator(masks_tif):
        masks.append(np.array(m))
    np_masks = np.array(masks)

    train_volume = np_volume[:28]
    train_masks = np_masks[:28]

    create_tf_record(os.path.join(main_tfrecords_dir,'ISBI_train.tfrecords'), train_volume, train_masks)

    val_volume = np_volume[28:]
    val_masks = np_masks[28:]
    create_tf_record(os.path.join(main_tfrecords_dir,'ISBI_val.tfrecords'), val_volume, val_masks)

if __name__ == '__main__':
    main_data_dir = '/media/data_cifs/andreas/connectomics/ISBI_2012_data/'
    main_tfrecords_dir = '/media/data_cifs/andreas/connectomics/tfrecords/'

    build_train_val_tfrecords(os.path.join(main_data_dir,'train'),main_tfrecords_dir)