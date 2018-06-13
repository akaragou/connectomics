#!/usr/bin/env python
from __future__ import division
import numpy as np
import os
import cv2
import h5py
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import random
from skimage.filters import scharr

def check_volume(data):
    """Ensure that data is numpy 3D array."""
    assert isinstance(data, np.ndarray)

    if data.ndim == 2:
        data = data[np.newaxis,...]
    elif data.ndim == 3:
        pass
    elif data.ndim == 4:
        assert data.shape[0]==1
        data = np.reshape(data, data.shape[-3:])
    else:
        raise RuntimeError('data must be a numpy 3D array')

    assert data.ndim==3
    return data

def affinitize(img, dst, dtype='float32'):
    """
    Transform segmentation to 3D affinity graph.
    Args:
        img: 3D indexed image, with each index corresponding to each segment.
    Returns:
        ret: affinity graph 
    """
    img = check_volume(img)
    ret = np.zeros((1,) + img.shape, dtype=dtype)

    (dz,dy,dx) = dst

    
    if dz != 0:
        # z-affinity.
        assert dz and abs(dz) < img.shape[-3]
        if dz > 0:
            ret[0,dz:,:,:] = (img[dz:,:,:]==img[:-dz,:,:]) & (img[dz:,:,:]>0)
        else:
            dz = abs(dz)
            ret[0,:-dz,:,:] = (img[dz:,:,:]==img[:-dz,:,:]) & (img[dz:,:,:]>0)

    if dy != 0:
        # y-affinity.
        assert dy and abs(dy) < img.shape[-2]
        if dy > 0:
            ret[0,:,dy:,:] = (img[:,dy:,:]==img[:,:-dy,:]) & (img[:,dy:,:]>0)
        else:
            dy = abs(dy)
            ret[0,:,:-dy,:] = (img[:,dy:,:]==img[:,:-dy,:]) & (img[:,dy:,:]>0)

    if dx != 0:
        # x-affinity.
        assert dx and abs(dx) < img.shape[-1]
        if dx > 0:
            ret[0,:,:,dx:] = (img[:,:,dx:]==img[:,:,:-dx]) & (img[:,:,dx:]>0)
        else:
            dx = abs(dx)
            ret[0,:,:,:-dx] = (img[:,:,dx:]==img[:,:,:-dx]) & (img[:,:,dx:]>0)

    return np.squeeze(ret)

def create_boundary_map(np_volume, np_masks, plot=False):
    adjusted_masks = []
    for i in range(np.shape(np_masks)[0]):
        mask = np.squeeze(np_masks[i,:,:])
        adjusted_mask = scharr(mask)
        adjusted_mask[adjusted_mask > 0.0000001] = 255
        adjusted_mask[adjusted_mask <= 0.0000001] = 0
        adjusted_mask = adjusted_mask.astype(np.uint8)
        idx = np.where((mask ==0 ))
        adjusted_mask[idx] = 255
        kernel = np.ones((3,3),np.uint8)
        adjusted_mask = cv2.dilate(adjusted_mask,kernel,iterations = 1)
        adjusted_mask = np.invert(adjusted_mask)
        adjusted_masks.append(adjusted_mask)
        if plot:
            f, (ax1,ax2,ax3) = plt.subplots(1,3)
            ax1.imshow(np.squeeze(np_volume[i,:,:]), cmap='gray')
            ax1.set_title('image')
            ax2.imshow(mask)
            ax2.set_title('labels')
            ax3.imshow(adjusted_mask, cmap='gray')
            ax3.set_title('binary mask')
            plt.show()

    adjusted_masks = np.array(adjusted_masks)
    return adjusted_mask

def random_crop_volume(volume,crop_size):

    D, H, W = np.shape(volume)
    d_crop = crop_size[0]
    h_crop = crop_size[1]
    w_crop = crop_size[2]

    d_index = random.randint(0, (D-d_crop))
    h_index = random.randint(0, (H-h_crop))
    w_index = random.randint(0, (W-w_crop))

    return volume[d_index:d_index+d_crop,h_index:h_index+h_crop,w_index:w_index+w_crop], (d_index, h_index, w_index)

def crop_volume(volume,crop_size,indicies):

    D, H, W = np.shape(volume)
    d_crop = crop_size[0]
    h_crop = crop_size[1]
    w_crop = crop_size[2]

    d_index = indicies[0]
    h_index = indicies[1]
    w_index = indicies[2]

    return volume[d_index:d_index+d_crop,h_index:h_index+h_crop,w_index:w_index+w_crop]


def load_images_and_labels(data_filepath, plot=False):
   
    volume = []
    volume_tif = Image.open(os.path.join(data_filepath,'train-input.tif'))
    for img in ImageSequence.Iterator(volume_tif):
        volume.append(np.array(img))
    volume = np.array(volume)
    volume = np.divide(volume, 255.0)

    seg = []
    seg_tif = Image.open(os.path.join(data_filepath,'train-labels.tif'))
    for m in ImageSequence.Iterator(seg_tif):
        seg.append(np.array(m))
    seg = np.array(seg).astype(np.uint8)

    distances = [  
                   (0,0,1),
                   (0,1,0),
                   (1,0,0),
                   (0,0,3),
                   (0,3,0),
                   (2,0,0),
                   (0,0,9),
                   (0,9,0),
                   (3,0,0),
                   (0,0,27),
                   (0,27,0),
                   (4,0,0)
                ]

    ground_truth_affities = []
    for i in range(len(distances)):
        aff = affinitize(seg, dst=distances[i])
        ground_truth_affities.append(aff.astype(int))

    ground_truth_affities = np.array(ground_truth_affities)

    if plot:
        for i in range(len(distances)):
            plt.imshow(np.squeeze(ground_truth_affities[i,80,:,:]), cmap='gray')
            plt.show()
        
    return volume, np.squeeze(ground_truth_affities[0])



if __name__ == '__main__':
    main_data_dir = '/media/data_cifs/andreas/connectomics/ISBI_2013_data/train'
    load_images_and_labels(main_data_dir)
    