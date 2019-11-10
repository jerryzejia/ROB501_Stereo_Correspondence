import numpy as np
from numpy.linalg import inv
from scipy.ndimage.filters import *
import time
def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive).
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond rng)

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il, greyscale.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Don't optimize for runtime (too much), optimize for clarity.

    #--- FILL ME IN ---

    # Your code goes here.
    
    #------------------
    '''
    Paper reference: R Zabih, J Woodfill (1994) ‎
    Non-parametric Local Transforms for Computing Visual Correspondence
    URL: http://www.cs.cornell.edu/~rdz/Papers/ZW-ECCV94.pdf

    Idea: Use non parametric transformation to create a kernel for each pixel. 

    '''
    h, w = Il.shape
    Id = np.zeros((h,w))
    window_size = 5
    
    left, right, up, down = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
    
    # Image transformation
    Il_transform, Ir_transform = image_transformation(Il, window_size), image_transformation(Ir, window_size)

    # Get score
    window_size = 3
    min_sad = np.full(Il.shape, float('inf'), dtype=np.float32)
    kernel = np.ones((window_size,window_size))
    
    for y in range(up, down):
        for x in range(left + window_size, right + window_size):
            image_patch_L = Il_transform[y - window_size:y+window_size, x - window_size : x + window_size]
            min_score = float("inf")
            min_d = None
            for d in range(maxd):
                l = x - d - window_size
                r = x - d + window_size
                l, r = boundary_ensurance(Il, l, r, window_size)
                image_patch_R = Ir_transform[y - window_size:y+window_size, l:r]
                score = np.sum(image_patch_R != image_patch_L)
                if score < min_score:
                    min_score = score
                    min_d = d
            Id[y, x] = min_d

    Id = gaussian_filter(Id, sigma = 1)

    return Id

def create_kernel(window_size, i): 
    kernel = np.zeros(window_size**2)
    kernel[i] = 0 if i == (window_size**2 - 1)//2 else 1
    kernel = kernel.reshape(window_size, window_size)
    return kernel

def image_transformation(I, window_size, mode = 'nearest'):
    """
    Create a transformation of the image
    Take the centre of the image patch as a threshold
    Transform everything bigger than the threshold into one, and everything lower as 0
    """
    h, w = I.shape
    depth = window_size**2 - 1 
    I_transform = np.zeros((h, w, depth))
    for i in range(depth):
        kernel = create_kernel(window_size, i)
        I_transform[:, :, i] = convolve(I, kernel, mode = mode, cval = 0)
    I_transform = (I_transform > np.stack([I]*depth, axis=-1)).astype(np.int)
    return I_transform

def boundary_ensurance(Il, l, r, window_size):
    h, w = Il.shape
    if r > w: 
        r = w
        l = r - 2 * window_size
    return l, r