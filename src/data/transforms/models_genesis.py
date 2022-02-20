""" Module models_genesis.py (By: Charley Zhang, Feb 2022)

Both 2D & 3D augmentations used by Models Genesis. Works only on numpy images.
Adapted from:
    https://github.com/MrGiovanni/ModelsGenesis/blob/master/pytorch/utils.py
"""

import math
import os
import random
import copy
import scipy
import imageio
import string
import numpy as np
from skimage.transform import resize
try:  # SciPy >= 0.19
    from scipy.special import comb  
except ImportError:
    from scipy.misc import comb
    
    
def flip(x, y, p=0.5):
    """
    2D and 3D flipping for grayscale images. 
    Args:
        x (np.ndarray): HxW or DxHxW image
        y (np.ndarray): HxW or DxHxW image
    """
    assert x.shape == y.shape
    
    ndims = x.ndim
    flipped_flags = [False] * ndims
    for axis in range(ndims):
        if random.random() < p:
            flipped_flags[axis] = True
            x = np.flip(x, axis=axis)
            y = np.flip(y, axis=axis)

    return x, y, flipped_flags


def local_pixel_shuffle(x, p=0.5, n_shuffle_windows=10000, 
                        max_window_size_ratio=0.1):
    """
    2D and 3D flipping for grayscale images. 
    Args:
        x (np.ndarray): HxW or DxHxW grayscale image
        n_shuffle_windows (int): number of windows to sample & shuffle
        max_window_size_raiot [0,1]: shuffle window size sampled between 0
            and ratio * length_of_dim (e.g. 64x64 patch, max window is 6x6
            with a ratio=0.1)
    """
    if random.random() >= p:
        return x
    
    ndims = x.ndim
    out_image = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    
    if ndims == 2:
        H, W = x.shape
        
        for _ in range(n_shuffle_windows):
            window_size_x = random.randint(1, int(H * max_window_size_ratio))
            window_size_y = random.randint(1, int(W * max_window_size_ratio))
            lower_x = random.randint(0, H - window_size_x)
            lower_y = random.randint(0, W - window_size_y)
            window = orig_image[lower_x:lower_x+window_size_x, 
                                lower_y:lower_y+window_size_y].flatten()
            np.random.shuffle(window)
            window = window.reshape((window_size_x, 
                                     window_size_y))
            out_image[lower_x:lower_x+window_size_x, 
                      lower_y:lower_y+window_size_y] = window
    else:
        assert ndims == 3
        D, H, W = x.shape
        
        for _ in range(n_shuffle_windows):
            window_size_x = random.randint(1, int(D * max_window_size_ratio))
            window_size_y = random.randint(1, int(H * max_window_size_ratio))
            window_size_z = random.randint(1, int(W * max_window_size_ratio))
            lower_x = random.randint(0, D - window_size_x)
            lower_y = random.randint(0, H - window_size_y)
            lower_z = random.randint(0, W - window_size_z)
            window = orig_image[lower_x:lower_x+window_size_x, 
                                lower_y:lower_y+window_size_y, 
                                lower_z:lower_z+window_size_z].flatten()
            np.random.shuffle(window)
            window = window.reshape((window_size_x, 
                                     window_size_y, 
                                     window_size_z))
            out_image[lower_x:lower_x+window_size_x, 
                      lower_y:lower_y+window_size_y, 
                      lower_z:lower_z+window_size_z] = window

    return out_image


def nonlinear_intensity_map(x, p=0.5):
    """
    Assumptions:
        1. x values must be between 0 and 1
        2. x has shapes HxW and DxHxW for 2D and 3D images, respectively 
    """
    if random.random() >= p:
        return x
    
    def bernstein_poly(i, n, t):
        """ Bernstein polynomial of n, i as a function of t. """
        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
    
    def bezier_curve(points, nTimes=1000):
        """  See: http://processingjs.nihongoresources.com/bezierinfo/
        Given set of control points, return bezier curve defined by control points.
        Control points should be a list of lists, or list of tuples
            such as [ [1,1], 
                      [2,3], 
                      [4,5], 
                      ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        """
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) 
                                     for i in range(0, nPoints)])        
        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals
    
    points = [[0, 0], 
              [random.random(), random.random()], 
              [random.random(), random.random()], 
              [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    
    if random.random() < 0.5:  # 50% chance of intensity get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals).astype(np.float32)
    return nonlinear_x


def in_paint(x, p=0.5, n_paints=5, uniform_paint=True):
    """ 
    Args:
        x (np.ndarray): HxW or DxHxW grayscale image
    """
    if random.random() >= p:
        return x

    if x.ndim == 2:
        H, W = x.shape
        while n_paints > 0 and random.random() < 0.95:
            block_noise_size_x = random.randint(H//6, H//3)
            block_noise_size_y = random.randint(W//6, W//3)
            noise_x = random.randint(3, H-block_noise_size_x-3)
            noise_y = random.randint(3, W-block_noise_size_y-3)
            
            patch_size = (block_noise_size_x, block_noise_size_y)
            if uniform_paint:
                uniform_paint_patch = np.ones(patch_size) * np.random.rand(1)[0]
            else:
                uniform_paint_patch = np.random.rand(patch_size) * 1.0
            x[noise_x:noise_x+block_noise_size_x, 
              noise_y:noise_y+block_noise_size_y] = uniform_paint_patch
            n_paints -= 1
    else:
        assert x.ndim == 3
        D, H, W = x.shape
        while n_paints > 0 and random.random() < 0.95:
            block_noise_size_x = random.randint(D//6, D//3)
            block_noise_size_y = random.randint(H//6, H//3)
            block_noise_size_z = random.randint(W//6, W//3)
            noise_x = random.randint(3, D-block_noise_size_x-3)
            noise_y = random.randint(3, H-block_noise_size_y-3)
            noise_z = random.randint(3, W-block_noise_size_z-3)
            
            patch_size = (block_noise_size_x, 
                          block_noise_size_y, 
                          block_noise_size_z)
            if uniform_paint:
                uniform_paint_patch = np.ones(patch_size) * np.random.rand(1)[0]
            else:
                uniform_paint_patch = np.random.rand(patch_size) * 1.0
            
            x[noise_x:noise_x+block_noise_size_x, 
              noise_y:noise_y+block_noise_size_y, 
              noise_z:noise_z+block_noise_size_z] = uniform_paint_patch
            n_paints -= 1
    return x


def out_paint(x, p=0.5, n_paints=4, uniform_paint=True):
    """ 
    Args:
        x (np.ndarray): HxW or DxHxW grayscale image
    """
    if random.random() >= p:
        return x

    image_temp = copy.deepcopy(x)
    if x.ndim == 2:
        H, W = x.shape
        if uniform_paint:
            x = np.ones((H, W)) * np.random.rand(1)[0]
        else:
            x = np.random.rand(H, W) * 1.0
        
        block_noise_size_x = H - random.randint(3*H//7, 4*H//7)
        block_noise_size_y = W - random.randint(3*W//7, 4*W//7)
        noise_x = random.randint(3, H-block_noise_size_x-3)
        noise_y = random.randint(3, W-block_noise_size_y-3)
        
        orig_patch = image_temp[noise_x:noise_x+block_noise_size_x, 
                                noise_y:noise_y+block_noise_size_y]
        x[noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y] = orig_patch
        
        while n_paints > 0 and random.random() < 0.95:
            block_noise_size_x = H - random.randint(3*H//7, 4*H//7)
            block_noise_size_y = W - random.randint(3*W//7, 4*W//7)
            noise_x = random.randint(3, H-block_noise_size_x-3)
            noise_y = random.randint(3, W-block_noise_size_y-3)
            
            orig_patch = image_temp[noise_x:noise_x+block_noise_size_x, 
                                    noise_y:noise_y+block_noise_size_y]
            x[noise_x:noise_x+block_noise_size_x, 
              noise_y:noise_y+block_noise_size_y] = orig_patch
            n_paints -= 1
    else:
        assert x.ndim == 3
        D, H, W = x.shape
        if uniform_paint:
            x = np.ones((D, H, W)) * np.random.rand(1)[0]
        else:
            x = np.random.rand(D, H, W) * 1.0
        
        block_noise_size_x = D - random.randint(3*D//7, 4*D//7)
        block_noise_size_y = H - random.randint(3*H//7, 4*H//7)
        block_noise_size_z = W - random.randint(3*W//7, 4*W//7)
        noise_x = random.randint(3, D-block_noise_size_x-3)
        noise_y = random.randint(3, H-block_noise_size_y-3)
        noise_z = random.randint(3, W-block_noise_size_z-3)
        
        orig_patch = image_temp[noise_x:noise_x+block_noise_size_x, 
                                noise_y:noise_y+block_noise_size_y, 
                                noise_z:noise_z+block_noise_size_z]
        x[noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = orig_patch
        
        while n_paints > 0 and random.random() < 0.95:
            block_noise_size_x = D - random.randint(3*D//7, 4*D//7)
            block_noise_size_y = H - random.randint(3*H//7, 4*H//7)
            block_noise_size_z = W - random.randint(3*W//7, 4*W//7)
            noise_x = random.randint(3, D-block_noise_size_x-3)
            noise_y = random.randint(3, H-block_noise_size_y-3)
            noise_z = random.randint(3, W-block_noise_size_z-3)
            
            orig_patch = image_temp[noise_x:noise_x+block_noise_size_x, 
                                    noise_y:noise_y+block_noise_size_y, 
                                    noise_z:noise_z+block_noise_size_z]
            x[noise_x:noise_x+block_noise_size_x, 
              noise_y:noise_y+block_noise_size_y, 
              noise_z:noise_z+block_noise_size_z] = orig_patch
            n_paints -= 1
    return x



    