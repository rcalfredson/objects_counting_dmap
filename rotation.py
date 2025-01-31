import cv2
import numpy as np


def split_by_channel(img):
    return tuple(img[..., i] for i in range(img.shape[-1]))


def to_int(tup):
    return tuple([int(el) for el in tup])


def rotate_image(mat, angle, use_linear=True):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2,
    )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(
        mat,
        rotation_mat,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR if use_linear else cv2.INTER_NEAREST,
    )
    return rotated_mat, (bound_h, bound_w)


def sample_patches(data, patch_size, num_patches):
    if len(patch_size) != data[0].ndim:
        raise ValueError()

    if not all((a.shape[:2] == data[0].shape[:2] for a in data)):
        raise ValueError(
            "all input shapes must be the same: %s"
            % (" / ".join(str(a.shape) for a in data))
        )

    if not all((0 < s <= d for s, d in zip(patch_size, data[0].shape))):
        raise ValueError(
            "patch_size %s negative or larger than data shape %s along some dimensions"
            % (str(patch_size), str(data[0].shape))
        )

    results = []
    # choose a random rotation angle
    for _ in range(num_patches):
        while True:
            rot_helper = RotationHelper(data[0].shape[0], data[0].shape[1], patch_size)
            result = rot_helper.get_random_rot_patch(data)
            if np.count_nonzero(result[0]) != 0:
                results.append(result)
                break
    return results


def rotate_point(pt, center, angle):
    temp_x = pt[1] - center[1]
    temp_y = pt[0] - center[0]
    rotated_x = temp_x * np.cos(angle) - temp_y * np.sin(angle)
    rotated_y = temp_x * np.sin(angle) + temp_y * np.cos(angle)
    return (rotated_y + center[0], rotated_x + center[1])


class RotationHelper:
    def __init__(self, height, width, patch_size):
        self.height = height
        self.width = width
        self.angle, self.angleRad = 0, 0
        self.patch_size = patch_size

    def get_rotation_angle(self):
        while True:
            self.angle = np.random.randint(0, 360)
            self.angleRad = np.pi * self.angle / 180
            self.aabb_h, self.aabb_w = self.calc_aabb_height_width()
            self.ht_range = self.height - self.aabb_h
            self.wd_range = self.width - self.aabb_w
            if self.ht_range > 0 and self.wd_range > 0:
                self.aabb_corner = (
                    np.random.uniform(0, self.height - self.aabb_h),
                    np.random.uniform(0, self.width - self.aabb_w),
                )
                self.aabb_center = (
                    self.aabb_corner[0] + 0.5 * self.aabb_h,
                    self.aabb_corner[1] + 0.5 * self.aabb_w,
                )
                break

    def get_random_rot_patch(self, data):
        self.get_rotation_angle()
        corner_point = rotate_point(
            (
                self.aabb_center[0] - 0.5 * self.patch_size[0],
                self.aabb_center[1] - 0.5 * self.patch_size[1],
            ),
            self.aabb_center,
            self.angleRad,
        )

        rotated_image = rotate_image(data[1], self.angle)[0]
        rotated_mask, bounds_post_rotation = rotate_image(data[0], self.angle)
        rotated_corner_point = rotate_point(
            corner_point, (self.height / 2, self.width / 2), -self.angleRad
        )
        rotated_corner_point = list(rotated_corner_point)
        rotated_corner_point[0] += 0.5 * (bounds_post_rotation[0] - self.height)
        rotated_corner_point[1] += 0.5 * (bounds_post_rotation[1] - self.width)
        rcp = rotated_corner_point
        sub_mask = rotated_mask[
            int(rcp[0]) : int(rcp[0] + self.patch_size[0]),
            int(rcp[1]) : int(rcp[1] + self.patch_size[1]),
        ]
        sub_img = rotated_image[
            int(rcp[0]) : int(rcp[0] + self.patch_size[0]),
            int(rcp[1]) : int(rcp[1] + self.patch_size[1]),
        ]
        return np.expand_dims(sub_mask, axis=0), np.moveaxis(sub_img, -1, 0)

    def calc_aabb_height_width(self):
        return (
            self.patch_size[0] * np.abs(np.cos(self.angleRad))
            + self.patch_size[1] * np.abs(np.sin(self.angleRad)),
            self.patch_size[0] * np.abs(np.sin(self.angleRad))
            + self.patch_size[1] * np.abs(np.cos(self.angleRad)),
        )
