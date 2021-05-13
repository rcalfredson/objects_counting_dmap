from data_loader import rotateImg
from rotation import rotate_point
import cv2
import numpy as np

def resize(img):
    return cv2.resize(img, (0,0), fx=2, fy=2)

PATCH_UBOUND = 160
patch = np.zeros((PATCH_UBOUND, PATCH_UBOUND, 1))
for i in range(4):
    y_loc, x_loc = [np.random.randint(0, PATCH_UBOUND) for _ in range(2)]
    print("Point %i coords: %i\t%i" % (i + 1, x_loc, y_loc))
    patch[y_loc, x_loc, :] = 1
# get coordinates of points from the image?
coords_of_pts = zip(*np.where(patch)[0:2])
rotation_angle = np.random.uniform(-90, 90)
print("rotation angle:", rotation_angle)
rotated_img = rotateImg(patch, rotation_angle, borderValue=(0), useInt=False)
reconstructed_patch = np.zeros(patch.shape)
for coords_of_pt in coords_of_pts:
    print("original point:", coords_of_pt)
    new_pt = rotate_point(
        coords_of_pt, [ln / 2 for ln in patch.shape[:2]], -np.pi * rotation_angle / 180
    )
    if all([coord < PATCH_UBOUND for coord in new_pt]):
        reconstructed_patch[int(new_pt[0]), int(new_pt[1])] = 1
    print("rotated point:", new_pt)
cv2.imshow("original", resize(patch))
cv2.imshow("rotated image", resize(rotated_img))
cv2.imshow("rotated points", resize(reconstructed_patch))
cv2.waitKey(0)