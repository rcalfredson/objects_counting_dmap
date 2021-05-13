import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter

patch = np.zeros((10, 10, 1))
patch[1, 1] = 100
filtered = gaussian_filter(patch, sigma=(1, 1, 0), order=0, mode='constant')
partial_filt = filtered[:, :]
# cv2.imshow("filtered", cv2.resize(partial_filt, (0, 0), fx=4, fy=4))
print("sum:", np.sum(partial_filt))
print(partial_filt)
print('max val:', np.amax(partial_filt))
plt.imshow(partial_filt)
plt.show()
# cv2.waitKey(0)

