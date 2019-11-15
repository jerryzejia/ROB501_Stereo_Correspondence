import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from imageio import imread
from stereo_disparity_fast import stereo_disparity_fast

# Load the stereo images.
<<<<<<< HEAD
Il = imread("cones_image_02.png", as_gray = True)
Ir = imread("cones_image_06.png", as_gray = True)
=======
Il = imread("teddy_image_02.png", as_gray = True)
Ir = imread("teddy_image_06.png", as_gray = True)
>>>>>>> 86720354be37b707ab15c68e58709feca6260ddd

# Load the appropriate bounding box.
bboxes = loadmat("bboxes.mat")
bbox = np.array(bboxes["teddy_02"]["bbox"])

Id = stereo_disparity_fast(Il, Ir, bbox, 52)
plt.imshow(Id, cmap = "gray")
plt.show()