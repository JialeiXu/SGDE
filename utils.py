from scipy.spatial.transform import Rotation as scipy_R
from Camera import Project_depth,BackprojectDepth
import copy
import cv2
import torch,time
import numpy as np
import torch.nn.functional as F
import os,wandb
import hashlib
import zipfile
from six.moves import urllib
import json,random
import shutil
import math
import matplotlib
import matplotlib.cm

def norm(vector):
    return math.sqrt(vector[0]*vector[0]+vector[1]*vector[1]+vector[2]*vector[2])


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6



def rotationMatrixToEulerAngles(R):
    r = scipy_R.from_matrix(R)
    res = r.as_rotvec()
    return res #for ceres

def gray_to_colormap(img,cmap='rainbow',max=None):
    '''
    Transfer gray map to matplotlib colormap
    '''
    assert  img.ndim == 2

    img[img<0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img / (max + 1e-8)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map  = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:,:,:3]*255).astype(np.uint8)
    colormap[mask_invalid] = 0
    return colormap

if __name__=='__main__':
    pass