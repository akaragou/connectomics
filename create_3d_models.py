#!/usr/bin/env python
from __future__ import division
import numpy as np
import os
import h5py
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from tqdm import tqdm

def build_3Dmodels(file_path):

    f = h5py.File(file_path, 'r')

    volume =  f['volume'][:].astype('uint8')
    masks =  f['masks'][:].astype('uint8')
    
    flatten_masks = masks.flatten()
    possible_ids = np.unique(flatten_masks)
    possible_ids = possible_ids.tolist()

    possible_ids.remove(0) # remove membranes

    sampled_ids = random.sample(possible_ids, 3) # sample 3 random neurons
    print sampled_ids

    neurons_to_plot = np.zeros(np.shape(masks))

    for z in tqdm(range(np.shape(masks)[0])):
        for h in range(np.shape(masks)[1]):
            for w in range(np.shape(masks)[2]):
                if masks[z,h,w] in sampled_ids:
                    neurons_to_plot[z,h,w] = masks[z,h,w] 

    verts, faces, normals, values = measure.marching_cubes_lewiner(neurons_to_plot, step_size=1, allow_degenerate=False)

    print np.shape(verts)
    print np.shape(values) 

    with open('neurons' + '_' + str(sampled_ids[0]) + '_' + str(sampled_ids[1]) + '_'+ str(sampled_ids[2])  + '.obj', 'w+') as f:
        for v in verts:
            f.write('v ' + str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n')
        for n in normals:
            f.write('vn ' + str(n[0]) + ' ' + str(n[1]) + ' ' + str(n[2]) + '\n')
        for fa in faces:
            f.write('f ' + \
                    str(fa[0] + 1) + '//' + str(fa[0] + 1) + ' ' + \
                    str(fa[1] + 1) + '//' + str(fa[1] + 1) + ' ' + \
                    str(fa[2] + 1) + '//' + str(fa[2] + 1) + '\n')


if __name__ == '__main__':
   build_3Dmodels('updated_Berson.h5')