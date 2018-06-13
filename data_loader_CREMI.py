#!/usr/bin/env python
from __future__ import division
import numpy as np
import os
import cv2
import h5py
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
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

def load_images_and_labels(data_filepath, plot=False):
   
    f = h5py.File(os.path.join(data_filepath,'sample_A_20160501.hdf'), 'r')
    print f.keys()

    volume =  f['volumes/raw'][:]
    seg =  f['volumes/labels/neuron_ids'][:]

    # nearest neighbor affinities
    distances = [  
                   (0,0,1),
                   (0,1,0),
                   (1,0,0),
                ]

    ground_truth_affities = []
    for i in range(len(distances)):
        aff = affinitize(seg, dst=distances[i])
        ground_truth_affities.append(aff.astype(int))

    ground_truth_affities = np.array(ground_truth_affities)

    # plotting x, y, z affinities for 80th slice
    if plot:
        for i in range(len(distances)):
            plt.imshow(np.squeeze(ground_truth_affities[i,80,:,:]), cmap='gray') 
            plt.show()
        
    return volume, np.squeeze(ground_truth_affities[0])

    
if __name__ == '__main__':
    main_data_dir = '/media/data_cifs/andreas/connectomics/CREMI_data/train'
    load_images_and_labels(main_data_dir, True)