import numpy as np
from numpy.linalg import inv
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.
    
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
    print(bbox)
    left, right, up, down = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
    window_size = 8
    # Initialize Disparity Map same size as IL
    Id = np.zeros((Il.shape[0], Il.shape[1]))
    for x in range(left, right):
        for y in range(up, down):
            for disparity in range(maxd):
                image_patch_A = None 
                image_patch_B = None 
                SAD = sum_of_absolute_difference(image_patch_A, image_patch_B)



    return Id

def sum_of_absolute_difference(image_patch_A, image_patch_B):
    # Load images and ravel in to flat array
    a = np.array(image_patch_A).ravel()
    b = np.array(image_patch_B).ravel()
    # Calculate the sum of absolute difference
    SAD = np.sum(np.abs(np.subtract(a,b,dtype=np.float)))
    return SAD

