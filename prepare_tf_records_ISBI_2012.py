#!/usr/bin/env python
from __future__ import division
import tensorflow as tf
import glob
import numpy as np
import os
from PIL import Image, ImageSequence
from tf_record import create_tf_record
import matplotlib.pyplot as plt

def build_tfrecords(data_filepath, tfrecords_filepath):
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

    train_volume = np_volume[:27]
    train_masks = np_masks[:27]
    # create_tf_record(os.path.join(main_tfrecords_dir,'ISBI_train.tfrecords'), train_volume, train_masks)

    n = len(np.ravel(train_volume))
    print np.sum(train_volume)/n

    val_volume = np_volume[27:28]
    val_masks = np_masks[27:28]
    # create_tf_record(os.path.join(main_tfrecords_dir,'ISBI_val.tfrecords'), val_volume, val_masks)

    test_volume = np_volume[28:]
    test_masks = np_masks[28:]
    # create_tf_record(os.path.join(main_tfrecords_dir,'ISBI_test.tfrecords'), test_volume, test_masks)

    unet = np.load('unet.npy')
    fusionNet = np.load('fusionNet.npy')
    tiramisu = np.load('tiramisu.npy')

    axes = plt.subplots(5,2)
    print np.shape(axes[1])
    ax1, ax2 = axes[1][0]
    ax3, ax4 = axes[1][1]
    ax5, ax6 = axes[1][2]
    ax7, ax8 = axes[1][3]
    ax9, ax10 = axes[1][4]

    ax1.imshow(np.squeeze(test_volume[0]), cmap='gray')
    ax1.set_title('Test Image 1')
    ax1.axis('off')

    ax2.imshow(np.squeeze(test_volume[1]), cmap='gray')
    ax2.set_title('Test Image 2')
    ax2.axis('off')

    ax3.imshow(np.squeeze(test_masks[0]), cmap='gray')
    ax3.set_title('Test Mask 1')
    ax3.axis('off')

    ax4.imshow(np.squeeze(test_masks[1]), cmap='gray')
    ax4.set_title('Test Mask 2')
    ax4.axis('off')

    ax5.imshow(np.squeeze(unet[0]), cmap='gray')
    ax5.set_title('Unet Prediction 1')
    ax5.axis('off')

    ax6.imshow(np.squeeze(unet[1]), cmap='gray')
    ax6.set_title('Unet Prediction 2')
    ax6.axis('off')

    ax7.imshow(np.squeeze(fusionNet[0]), cmap='gray')
    ax7.set_title('FusionNet Prediction 1')
    ax7.axis('off')

    ax8.imshow(np.squeeze(fusionNet[1]), cmap='gray')
    ax8.set_title('FusionNet Prediction 2')
    ax8.axis('off')

    ax9.imshow(np.squeeze(tiramisu[0]), cmap='gray')
    ax9.set_title('Tiramisu Prediction 1')
    ax9.axis('off')

    ax10.imshow(np.squeeze(tiramisu[1]), cmap='gray')
    ax10.set_title('Tiramisu Prediction 2 ')
    ax10.axis('off')
    plt.show()


if __name__ == '__main__':
    main_data_dir = '/media/data_cifs/andreas/connectomics/ISBI_2012_data/'
    main_tfrecords_dir = '/media/data_cifs/andreas/connectomics/tfrecords/'

    build_tfrecords(os.path.join(main_data_dir,'train'),main_tfrecords_dir)