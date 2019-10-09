import numpy as np
import scipy.ndimage
from scipy import interpolate, stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os
import array
import cv2

#readin Image
def readin_vanhateren_img(fname):
    '''
    Readin a single van hateren image
    '''

    with open(fname, 'rb') as handle:
       s = handle.read()
    arr = array.array('H', s)
    arr.byteswap()
    img = np.array(arr, dtype='uint16').reshape(1024, 1536)
    return(img)


#readin Image
def readin_jpg_img(fname):
    '''
    Readin a single em image
    '''
    s = cv2.imread(fname)
    s = np.mean(s,axis=-1)#mean to greyscale
    return(np.array(s))

def readin_img(fname, imtype):
    if(imtype=='vanhateren'):
        return(readin_vanhateren_img(fname))
    elif(imtype in ['em','astro']):
        return(readin_jpg_img(fname))
    else:
        raise ValueError(f'imtype of {imtype} not supported.')