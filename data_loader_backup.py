"""PyTorch dataset for HDF5 files generated with `get_data.py`."""
import os
import random
from rotation import rotate_point

import cv2
import h5py
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage.feature.peak import peak_local_max
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def rotateImg(
    image, angle, center=None, scale=1.0, borderValue=(255, 255, 255), useInt=True
):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=borderValue
    )

    return rotated / (255 if useInt else 1)


def background_color(img):
    if type(img) != Image.Image:
        pil_img = Image.fromarray(img)
    else:
        pil_img = img
    return max(pil_img.getcolors(pil_img.size[0] * pil_img.size[1]))[1]


def rotate_label(label, rotationAngle):
    coords_of_pts = zip(*np.where(label)[0:2])
    reconstructed_label = np.zeros(label.shape)
    for pt in coords_of_pts:
        new_pt = rotate_point(
            pt, [ln / 2 for ln in label.shape[:2]], -np.pi * rotationAngle / 180
        )
        rounded_coords = [round(coord) for coord in new_pt]
        if all([coord < label.shape[i] and coord >=0 for i, coord in enumerate(rounded_coords)]):
            reconstructed_label[rounded_coords[0], rounded_coords[1]] = 100
    return reconstructed_label


def rotateInCollateStep(imagePatch, labelPatch):
    rotationAngle = np.random.uniform(-180, 180)
    imagePatch = np.multiply(imagePatch, 255).T.astype(np.uint8)
    backgnd_color = background_color(imagePatch)
    rotatedImg = rotateImg(
        imagePatch, rotationAngle, borderValue=backgnd_color, useInt=True
    )
    # rotatedLabel = rotateImg(
    #     labelPatch, rotationAngle, borderValue=(0, 0, 0), useInt=False
    # )
    rotatedLabel = rotate_label(labelPatch, rotationAngle)
    # rotatedLabel = np.expand_dims(rotatedLabel, axis=-1)
    rotatedImg = rotatedImg.astype(np.float32)
    return rotatedImg.T, rotatedLabel


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty : starty + cropy, startx : startx + cropx]


def augment_with_zoom(img, lbl, zoom_level, debug=False):
    img = img.T
    lbl = np.squeeze(lbl)
    resized_img = cv2.resize(img, (0, 0), fx=zoom_level, fy=zoom_level)
    resized_lbl = cv2.resize(lbl, (0, 0), fx=zoom_level, fy=zoom_level)
    peak_positions = peak_local_max(
        resized_lbl, min_distance=2, threshold_abs=0.78, indices=False
    ).astype(np.float32)
    if debug:
        cv2.imshow("orig img", img)
        cv2.imshow("orig lbl", lbl)
        cv2.imshow("resized", resized_lbl)
    resized_lbl = 100 * peak_positions
    resized_lbl = gaussian_filter(resized_lbl, sigma=(1, 1), order=0)
    if debug:
        cv2.imshow("resized img:", resized_img)
        cv2.imshow("resampled:", resized_lbl)
        print("zoom level:", zoom_level)
        cv2.waitKey(0)
    return resized_img.T, resized_lbl


def chooseRandomRegions(
    batch,
    numPatches,
    size,
    shuffle=False,
    skipEmpties=False,
    randomPatchSizes=False,
    zoomMin=None,
    zoomMax=None,
    applyGaussian=False,
    debug=False,
):
    useZoom = zoomMin is not None and zoomMax is not None
    zoom_level = 1.0

    def refresh_zoom_level():
        nonlocal zoom_level
        if not useZoom:
            return
        zoom_level = np.random.uniform(zoomMin, zoomMax)

    if useZoom:
        refresh_zoom_level()
    if randomPatchSizes and np.random.uniform() < 0.1:
        # choose an image, regardless of whether it's empty
        img_idx = np.random.randint(0, len(batch))
        if useZoom:
            zoomed_img, zoomed_lbl = augment_with_zoom(
                batch[img_idx][0], batch[img_idx][1], zoom_level
            )
        else:
            zoomed_img, zoomed_lbl = batch[img_idx]
        img_ht = zoomed_img.shape[1]
        img_wd = zoomed_img.shape[-1]
        resizedImgs = torch.zeros((1, 3, img_ht, img_wd), dtype=torch.float32)
        resizedLabels = torch.zeros((1, 1, img_ht, img_wd), dtype=torch.float32)
        resizedImgs[0] = torch.from_numpy(zoomed_img)
        resizedLabels[0] = torch.from_numpy(zoomed_lbl)
        return resizedImgs, resizedLabels
    # determine patch size for this batch
    # find minimum height and width of all images
    if randomPatchSizes:
        ht_min, wd_min = 0, 0
        for el in batch:
            if ht_min == 0:
                ht_min = el[0].shape[1]
            if wd_min == 0:
                wd_min = el[0].shape[-1]
            if el[0].shape[1] < ht_min:
                ht_min = el[0].shape[1]
            if el[0].shape[-1] < wd_min:
                wd_min = el[0].shape[-1]
        patch_ht = np.random.randint(144, ht_min) * zoom_level
        patch_wd = np.random.randint(144, wd_min) * zoom_level
    else:
        patch_ht, patch_wd = size, size
    final_ht = np.round(patch_ht).astype(int)
    final_wd = np.round(patch_wd).astype(int)
    resizedImgs = torch.zeros(
        (len(batch) * numPatches, 3, final_ht, final_wd), dtype=torch.float32
    )
    resizedLabels = torch.zeros(
        (len(batch) * numPatches, 1, final_ht, final_wd), dtype=torch.float32
    )

    region_counter = 0
    for i, el in enumerate(batch):
        if skipEmpties and np.count_nonzero(el[1]) == 0:
            continue
        counter = 0
        while counter < numPatches:
            refresh_zoom_level()
            x_origin = np.random.randint(0, el[0].shape[-1] - patch_wd)
            y_origin = np.random.randint(0, el[0].shape[1] - patch_ht)
            label_patch = el[1][
                :, y_origin : y_origin + patch_ht, x_origin : x_origin + patch_wd
            ]
            if skipEmpties and np.count_nonzero(label_patch) == 0:
                continue
            img_patch = el[0][
                :, y_origin : y_origin + patch_ht, x_origin : x_origin + patch_wd
            ]
            label_patch = label_patch.T
            img_patch, label_patch = rotateInCollateStep(img_patch, label_patch)
            if applyGaussian:
                label_patch = gaussian_filter(
                    label_patch, sigma=(1, 1, 0), order=0, mode="reflect"
                )
            if useZoom:
                zoomed_patch, zoomed_label = augment_with_zoom(
                    img_patch, label_patch, zoom_level
                )
            else:
                zoomed_patch, zoomed_label = img_patch, label_patch

            if zoomed_patch.shape[1] > final_ht:
                zoomed_patch = zoomed_patch[:, :final_ht, :final_wd]
                zoomed_label = zoomed_label[:final_ht, :final_wd]
            elif zoomed_patch.shape[1] < final_ht:
                zoomed_patch = (zoomed_patch * 255).astype(np.uint8)
                bckgnd = background_color(zoomed_patch.T)
                zoomed_patch_new = np.full((size, size, 3), fill_value=bckgnd).T
                zoomed_patch_new[
                    :, : zoomed_patch.shape[1], : zoomed_patch.shape[-1]
                ] = zoomed_patch
                zoomed_patch = zoomed_patch_new.astype(np.float32) / 255
                zoomed_label_new = np.zeros((size, size))
                zoomed_label_new[
                    : zoomed_label.shape[0], : zoomed_label.shape[1]
                ] = zoomed_label
                zoomed_label = zoomed_label_new

            if debug:
                print("shape of zoomed patch:", zoomed_patch.shape)
                print("shape of zoomed label:", zoomed_label.shape)
                print("sum of zoomed label: %.15f" % (np.sum(zoomed_label) / 100.0))
                cv2.imshow("debug", cv2.resize(zoomed_patch.T, (0, 0), fx=2, fy=2))
                cv2.imshow("zoomed label", cv2.resize(zoomed_label, (0, 0), fx=2, fy=2)) # cv2.resize(zoomed_label, (0, 0), fx=2, fy=2))
                cv2.waitKey(0)
            resizedImgs[region_counter * numPatches + counter] = torch.from_numpy(
                zoomed_patch
            )

            resizedLabels[region_counter * numPatches + counter] = torch.from_numpy(
                zoomed_label.T
            )
            counter += 1
        region_counter += 1
    resizedImgs = resizedImgs[: region_counter * numPatches]
    resizedLabels = resizedLabels[: region_counter * numPatches]
    if shuffle:
        indexOrder = list(range(resizedImgs.shape[0]))
        random.shuffle(indexOrder)
        indexOrder = torch.LongTensor(indexOrder)
        resizedImgsOld = resizedImgs
        resizedLabelsOld = resizedLabels
        resizedImgs = torch.zeros_like(resizedImgsOld)
        resizedLabels = torch.zeros_like(resizedLabelsOld)
        resizedImgs = resizedImgsOld[indexOrder]
        resizedLabels = resizedLabelsOld[indexOrder]
    return resizedImgs, resizedLabels


def padImagesBasedOnMaxSize(batch):
    maxHt, maxWidth = 0, 0
    for el in batch:
        img = el[0]
        if img.shape[1] > maxHt:
            maxHt = img.shape[1]
        if img.shape[-1] > maxWidth:
            maxWidth = img.shape[-1]
    resizedImgs = torch.zeros((len(batch), 3, maxHt, maxWidth), dtype=torch.float32)
    resizedLabels = torch.zeros((len(batch), 1, maxHt, maxWidth), dtype=torch.float32)
    for i, el in enumerate(batch):
        asymmetryCorrs = {"vp": 0, "hp": 0}
        if el[0].shape[2] < maxWidth:
            dividend = (maxWidth - el[0].shape[2]) / 2
            asymmetryCorrs["hp"] = 1 if dividend % 1 > 0 else 0
            hp = int(dividend)
        else:
            hp = 0
        if el[0].shape[1] < maxHt:
            dividend = (maxHt - el[0].shape[1]) / 2
            asymmetryCorrs["vp"] = 1 if dividend % 1 > 0 else 0
            vp = int(dividend)
        else:
            vp = 0
        padding = (hp, hp + asymmetryCorrs["hp"], vp, vp + asymmetryCorrs["vp"])
        padded = np.multiply(el[0], 255).astype(np.uint8)
        bckgnd = background_color(padded)
        padded = torch.nn.functional.pad(
            torch.Tensor(padded), padding, mode="constant", value=bckgnd
        )
        resizedImgs[i] = torch.div(padded, 255.0)
        padded = torch.nn.functional.pad(
            torch.Tensor(el[1]), padding, mode="constant", value=0
        )
        resizedLabels[i] = padded

    return resizedImgs, resizedLabels


class DatasetWithAugs(Dataset):
    def __init__(
        self,
        horizontal_flip: float = 0.0,
        vertical_flip: float = 0.0,
        rand_rotate: bool = False,
        rand_brightness=None,
        debug=False,
    ):
        super(DatasetWithAugs, self).__init__()
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rand_rotate = rand_rotate
        self.rand_brightness = rand_brightness
        self.images = []
        self.labels = []
        self.debug = debug

    def __len__(self):
        """Return no. of samples in HDF5 file."""
        return len(self.images)

    def __randbrightness__(self, image):
        return image

    def __randrotate__(self, image, label):
        if True or not self.rand_rotate or image.shape[0] != image.shape[1]:
            return (np.divide(image.astype(np.float32).T, 255), label)

    def __getitem__(self, index: int):
        """Return next sample (randomly flipped)."""
        # if both flips probabilities are zero return an image and a label
        if not (self.horizontal_flip or self.vertical_flip):
            return self.images[index], self.labels[index]

        # axis = 1 (vertical flip), axis = 2 (horizontal flip)
        axis_to_flip = []

        if random.random() < self.vertical_flip:
            axis_to_flip.append(1)

        if random.random() < self.horizontal_flip:
            axis_to_flip.append(2)

        image = np.flip(self.images[index], axis=axis_to_flip).copy()

        image = (255 * image.T).astype(np.uint8)
        if self.rand_brightness:
            image = self.__randbrightness__(image)

        final_results = self.__randrotate__(
            image, np.flip(self.labels[index], axis=axis_to_flip).copy()
        )

        if self.debug:
            cv2.imshow("imgFInal", final_results[0].T)
            cv2.imshow("labelsFinal", final_results[1].T)
            cv2.waitKey(0)

        return final_results


class MultiDimH5Dataset(DatasetWithAugs):
    def __init__(
        self,
        dataset_path: str,
        horizontal_flip: float = 0.0,
        vertical_flip: float = 0.0,
        rand_rotate: bool = False,
        rand_brightness=None,
    ):
        """
        Initialize flips probabilities and pointers to a HDF5 file.

        Args:
            dataset_path: a path to a HDF5 file
            horizontal_flip: the probability of applying horizontal flip
            vertical_flip: the probability of applying vertical flip
        """
        super().__init__(horizontal_flip, vertical_flip, rand_rotate, rand_brightness)
        self.h5 = h5py.File(dataset_path, "r")
        self.images = list(
            set([keyName.replace("_dots", "") for keyName in self.h5.keys()])
        )
        self.labels = ["%s_dots" % imgName for imgName in self.images]
        self.horizontal_flip

    def __randrotate__(self, image, label):
        """Perform a random rotation on the image and labels."""
        if not self.rand_rotate or image.shape[0] != image.shape[1]:
            return (np.divide(image.astype(np.float32).T, 255), label)
        rotationAngle = np.random.uniform(-90, 90)
        backgnd_color = background_color(image)
        rotatedImg = rotateImg(image, rotationAngle, borderValue=backgnd_color)
        rotatedLabel = rotateImg(
            label.T, rotationAngle, borderValue=(0, 0, 0), useInt=False
        )
        rotatedLabel = np.expand_dims(rotatedLabel, axis=-1)
        rotatedImg = rotatedImg.astype(np.float32)

        return (rotatedImg.T, rotatedLabel.T)

    def __getitem__(self, index: int):
        """Return next sample (randomly flipped)."""
        # if both flips probabilities are zero return an image and a label
        if not (self.horizontal_flip or self.vertical_flip):
            return self.h5[self.images[index]][0], self.h5[self.labels[index]][0]

        # axis = 1 (vertical flip), axis = 2 (horizontal flip)
        axis_to_flip = []

        if random.random() < self.vertical_flip:
            axis_to_flip.append(1)

        if random.random() < self.horizontal_flip:
            axis_to_flip.append(2)

        image = np.flip(self.h5[self.images[index]][0], axis=axis_to_flip).copy()

        if self.rand_rotate:
            image = (255 * image.T).astype(np.uint8)
            final_results = self.__randrotate__(
                image, np.flip(self.h5[self.labels[index]][0], axis=axis_to_flip).copy()
            )
        else:
            final_results = (
                image,
                np.flip(self.h5[self.labels[index]][0], axis=axis_to_flip).copy(),
            )
        return final_results


class H5Dataset(DatasetWithAugs):
    """PyTorch dataset for HDF5 files generated with `get_data.py`."""

    def __init__(
        self,
        dataset_path: str,
        horizontal_flip: float = 0.0,
        vertical_flip: float = 0.0,
        rand_rotate: bool = False,
        rand_brightness: bool = False,
        debug=False,
    ):
        """
        Initialize flips probabilities and pointers to a HDF5 file.

        Args:
            dataset_path: a path to a HDF5 file
            horizontal_flip: the probability of applying horizontal flip
            vertical_flip: the probability of applying vertical flip
        """
        super().__init__(
            horizontal_flip, vertical_flip, rand_rotate, rand_brightness, debug=debug
        )
        self.h5 = h5py.File(dataset_path, "r")
        self.images = self.h5["images"]
        self.labels = self.h5["labels"]

    def __randrotate__(self, image, label):
        """Perform a random rotation on the image and labels."""
        if not self.rand_rotate:
            return (image.astype(np.float32).T, label)
        rotationAngle = np.random.uniform(-90, 90)
        backgnd_color = background_color(image)
        rotatedImg = rotateImg(image, rotationAngle, borderValue=backgnd_color)
        rotatedLabel = rotateImg(
            label.T, rotationAngle, borderValue=(0, 0, 0), useInt=False
        )
        rotatedLabel = np.expand_dims(rotatedLabel, axis=-1)
        rotatedImg = rotatedImg.astype(np.float32)

        return (rotatedImg.T, rotatedLabel.T)

    def __randbrightness__(self, image):
        value = np.random.uniform(0.5, 1.5)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img


# --- PYTESTS --- #


def test_loader():
    """Test HDF5 dataloader with flips on and off."""
    run_batch(flip=False)
    run_batch(flip=True)


def run_batch(flip):
    """Sanity check for HDF5 dataloader checks for shapes and empty arrays."""
    # datasets to test loader on
    datasets = {"cell": (3, 256, 256), "mall": (3, 480, 640), "ucsd": (1, 160, 240)}

    # for each dataset check both training and validation HDF5
    # for each one check if shapes are right and arrays are not empty
    for dataset, size in datasets.items():
        for h5 in ("train.h5", "valid.h5"):
            # create a loader in "all flips" or "no flips" mode
            data = H5Dataset(
                os.path.join(dataset, h5),
                horizontal_flip=1.0 * flip,
                vertical_flip=1.0 * flip,
            )
            # create dataloader with few workers
            data_loader = DataLoader(data, batch_size=4, num_workers=4)

            # take one batch, check samples, and go to the next file
            for img, label in data_loader:
                # image batch shape (#workers, #channels, resolution)
                assert img.shape == (4, *size)
                # label batch shape (#workers, 1, resolution)
                assert label.shape == (4, 1, *size[1:])

                assert torch.sum(img) > 0
                assert torch.sum(label) > 0

                break
