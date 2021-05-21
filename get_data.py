"""A tool to download and preprocess data, and generate HDF5 file.

Available datasets:

    * cell: http://www.robots.ox.ac.uk/~vgg/research/counting/index_org.html
    * mall: http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html
    * ucsd: http://www.svcl.ucsd.edu/projects/peoplecnt/
"""
from collections import Counter
import json
import os
import shutil
import zipfile
from glob import glob
from random import shuffle
from typing import List, Tuple

import click
import h5py
import wget
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["cell", "mall", "ucsd", "egg", "egg-test"]),
    required=True,
)
def get_data(dataset: str):
    """
    Get chosen dataset and generate HDF5 files with training
    and validation samples.
    """
    # dictionary-based switch statement
    {
        "cell": generate_cell_data,
        "mall": generate_mall_data,
        "ucsd": generate_ucsd_data,
        "egg": generate_egg_datasets,
        "egg-test": generate_egg_heldout_data,
    }[dataset]()


def create_hdf5(
    dataset_name: str,
    train_size: int,
    valid_size: int,
    img_size: Tuple[int, int],
    in_channels: int = 3,
    multiSize=False,
    train_name=None,
):
    """
    Create empty training and validation HDF5 files with placeholders
    for images and labels (density maps).

    Note:
    Datasets are saved in [dataset_name]/train.h5 and [dataset_name]/valid.h5.
    Existing files will be overwritten.

    Args:
        dataset_name: used to create a folder for train.h5 and valid.h5
        train_size: no. of training samples
        valid_size: no. of validation samples
        img_size: (width, height) of a single image / density map
        in_channels: no. of channels of an input image

    Returns:
        A tuple of pointers to training and validation HDF5 files.
    """
    # create output folder if it does not exist
    os.makedirs(dataset_name, exist_ok=True)
    if multiSize:
        if train_size > 0:
            if not train_name:
                train_name = "train.h5"
            trainDataFile = h5py.File(os.path.join(dataset_name, train_name), "w")
        else:
            trainDataFile = None
        validDataFile = h5py.File(os.path.join(dataset_name, "valid.h5"), "w")
        return trainDataFile, validDataFile
    # create HDF5 files: [dataset_name]/(train | valid).h5
    train_h5 = h5py.File(os.path.join(dataset_name, "train.h5"), "w")
    valid_h5 = h5py.File(os.path.join(dataset_name, "valid.h5"), "w")

    # add two HDF5 datasets (images and labels) for each HDF5 file
    for h5, size in ((train_h5, train_size), (valid_h5, valid_size)):
        h5.create_dataset("images", (size, in_channels, *img_size))
        h5.create_dataset("labels", (size, 1, *img_size))

    return train_h5, valid_h5


def generate_label(label_info: np.array, image_shape: List[int]):
    """
    Generate a density map based on objects positions.

    Args:
        label_info: (x, y) objects positions
        image_shape: (width, height) of a density map to be generated

    Returns:
        A density map.
    """
    # create an empty density map
    label = np.zeros(image_shape, dtype=np.float32)

    # loop over objects positions and marked them with 100 on a label
    # note: *_ because some datasets contain more info except x, y coordinates
    for x, y, *_ in label_info:
        if y < image_shape[0] and x < image_shape[1]:
            label[int(y)][int(x)] = 100

    # apply a convolution with a Gaussian kernel
    label = gaussian_filter(label, sigma=(1, 1), order=0)

    return label


def get_and_unzip(url: str, location: str = "."):
    """Extract a ZIP archive from given URL.

    Args:
        url: url of a ZIP file
        location: target location to extract archive in
    """
    dataset = wget.download(url)
    dataset = zipfile.ZipFile(dataset)
    dataset.extractall(location)
    dataset.close()
    os.remove(dataset.filename)


def generate_ucsd_data():
    """Generate HDF5 files for mall dataset."""
    # download and extract data
    get_and_unzip("http://www.svcl.ucsd.edu/projects/peoplecnt/db/ucsdpeds.zip")
    # download and extract annotations
    get_and_unzip("http://www.svcl.ucsd.edu/projects/peoplecnt/db/vidf-cvpr.zip")
    # create training and validation HDF5 files
    train_h5, valid_h5 = create_hdf5(
        "ucsd", train_size=1500, valid_size=500, img_size=(160, 240), in_channels=1
    )

    def fill_h5(h5, labels, video_id, init_frame=0, h5_id=0):
        """
        Save images and labels in given HDF5 file.

        Args:
            h5: HDF5 file
            labels: the list of labels
            video_id: the id of a scene
            init_frame: the first frame in given list of labels
            h5_id: next dataset id to be used
        """
        video_name = f"vidf1_33_00{video_id}"
        video_path = f"ucsdpeds/vidf/{video_name}.y/"

        for i, label in enumerate(labels, init_frame):
            # path to the next frame (convention: [video name]_fXXX.jpg)
            img_path = f"{video_path}/{video_name}_f{str(i+1).zfill(3)}.png"

            # get an image as numpy array
            image = np.array(Image.open(img_path), dtype=np.float32) / 255
            # generate a density map by applying a Gaussian filter
            label = generate_label(label[0][0][0], image.shape)

            # pad images to allow down and upsampling
            image = np.pad(image, 1, "constant", constant_values=0)
            label = np.pad(label, 1, "constant", constant_values=0)

            # save data to HDF5 file
            h5["images"][h5_id + i - init_frame, 0] = image
            h5["labels"][h5_id + i - init_frame, 0] = label

    # dataset contains 10 scenes
    for scene in range(10):
        # load labels infomation from provided MATLAB file
        # it is numpy array with (x, y) objects position for subsequent frames
        descriptions = loadmat(f"vidf-cvpr/vidf1_33_00{scene}_frame_full.mat")
        labels = descriptions["frame"][0]

        # use first 150 frames for training and the last 50 for validation
        # start filling from the place last scene finished
        fill_h5(train_h5, labels[:150], scene, 0, 150 * scene)
        fill_h5(valid_h5, labels[150:], scene, 150, 50 * scene)

    # close HDF5 files
    train_h5.close()
    valid_h5.close()

    # cleanup
    shutil.rmtree("ucsdpeds")
    shutil.rmtree("vidf-cvpr")


def generate_mall_data():
    """Generate HDF5 files for mall dataset."""
    # download and extract dataset
    get_and_unzip(
        "http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip"
    )
    # create training and validation HDF5 files
    train_h5, valid_h5 = create_hdf5(
        "mall", train_size=1500, valid_size=500, img_size=(480, 640), in_channels=3
    )

    # load labels infomation from provided MATLAB file
    # it is a numpy array with (x, y) objects position for subsequent frames
    labels = loadmat("mall_dataset/mall_gt.mat")["frame"][0]

    def fill_h5(h5, labels, init_frame=0):
        """
        Save images and labels in given HDF5 file.

        Args:
            h5: HDF5 file
            labels: the list of labels
            init_frame: the first frame in given list of labels
        """
        for i, label in enumerate(labels, init_frame):
            # path to the next frame (filename convention: seq_XXXXXX.jpg)
            img_path = f"mall_dataset/frames/seq_{str(i+1).zfill(6)}.jpg"

            # get an image as numpy array
            image = np.array(Image.open(img_path), dtype=np.float32) / 255
            image = np.transpose(image, (2, 0, 1))

            # generate a density map by applying a Gaussian filter
            label = generate_label(label[0][0][0], image.shape[1:])

            # save data to HDF5 file
            h5["images"][i - init_frame] = image
            h5["labels"][i - init_frame, 0] = label

    # use first 1500 frames for training and the last 500 for validation
    fill_h5(train_h5, labels[:1500])
    fill_h5(valid_h5, labels[1500:], 1500)

    # close HDF5 file
    train_h5.close()
    valid_h5.close()

    # cleanup
    shutil.rmtree("mall_dataset")


def get_egg_image_paths(held_out=False):
    all_images = glob(
        "egg_source/heldout_robert_task1%s/*_dots*" % ("" if held_out else "")
    )
    shuffle(all_images)
    return all_images


def get_egg_image_paths_combined(folder_list):
    all_images = []
    for folder in folder_list:
        all_images += glob("%s/*_dots*" % folder)
    shuffle(all_images)
    return all_images


def generate_egg_heldout_data():
    all_images = get_egg_image_paths(held_out=True)
    in_channels = 3
    _, valid_h5 = create_hdf5(
        "egg-eval-robert-task-1",
        train_size=0,
        valid_size=None,
        img_size=(None, None),
        in_channels=in_channels,
        multiSize=True,
    )
    for i, label_path in enumerate(all_images):
        labelImg = np.array(Image.open(label_path))
        basename = os.path.basename(label_path).replace("_dots.png", "")
        img_path = label_path.replace("_dots.png", ".jpg")
        valid_h5.create_dataset("%s_dots" % basename, (1, 1, *labelImg.shape[0:2]))
        valid_h5.create_dataset(basename, (1, in_channels, *labelImg.shape[0:2]))
        image = np.array(Image.open(img_path), dtype=np.float32) / 255
        image = np.transpose(image, (2, 0, 1))
        labelImg = 100.0 * (labelImg[:, :, 0] > 0)
        label = gaussian_filter(labelImg, sigma=(1, 1), order=0)
        valid_h5[basename][0] = image
        valid_h5["%s_dots" % basename][0, 0] = label


def generate_egg_datasets():
    # generate_egg_data(fullSize=True)
    generate_egg_data(mode='fullSize', annot_style='point')
    # generate_egg_data(mode='patches')
    # generate_egg_data(mode='from_file')
    # generate_egg_data(mode="multiple_folders")


def generate_egg_data(mode="patches", annot_style='gaussian'):
    """Generate HDF5 files for egg-laying images."""
    multiSize = mode in ("fullSize", "combined", "multiple_folders")
    if mode != "patches":
        datasetSuffix = "-%s" % mode.lower()
    else:
        datasetSuffix = ""
    # image names need to be sourced from both fullsize and patch directories
    fullSizeDirs = [
        "egg_source/archive_2021-03-22/fullsize_%s/*_dots*" % setType
        for setType in ("train", "valid")
    ]
    if mode == "patches":
        training_image_names = glob("egg_source/archive_2021-03-22/train/*_dots*")
        # temporary downsizing of the dataset
        # training_image_names = list(np.random.choice(training_image_names, 1500, replace=False))
        validation_image_names = glob("egg_source/archive_2021-03-22/valid/*_dots*")
        # validation_image_names = list(np.random.choice(validation_image_names, int(0.25*1500), replace=False))
    elif mode == "fullSize":
        training_image_names = glob(fullSizeDirs[0])
        validation_image_names = glob(fullSizeDirs[1])
    elif mode == "combined":
        with open("egg_source/trainPatchFullCombinedImgList.txt", "r") as f:
            training_image_names = f.read().splitlines()
        with open("egg_source/validPatchFullCombinedImgList.txt", "r") as f:
            validation_image_names = f.read().splitlines()
        training_image_names = list(
            np.random.choice(
                training_image_names,
                int(0.25 * len(training_image_names)),
                replace=False,
            )
        )
        validation_image_names = list(
            np.random.choice(
                validation_image_names,
                int(0.25 * len(validation_image_names)),
                replace=False,
            )
        )
    elif mode == "from_file":
        with open(r"C:\Users\Tracking\splinedist\comparison\train.json", "r") as f:
            training_image_names = json.load(f)
        with open(r"C:\Users\Tracking\splinedist\comparison\valid.json", "r") as f:
            validation_image_names = json.load(f)
    elif mode == "multiple_folders":
        training_image_names = get_egg_image_paths_combined(
            [
                r"P:\Robert\objects_counting_dmap\egg_source"
                + r"\heldout_robert_WT_5"
            ]
        )
        validation_image_names = []
        print("how many training image names?", len(training_image_names))

    train_h5, valid_h5 = create_hdf5(
        "eggTemp%s" % datasetSuffix,
        train_size=len(training_image_names),
        valid_size=len(validation_image_names),
        img_size=(160, 160),
        in_channels=3,
        multiSize=multiSize,
        train_name=None,
    )

    def fill_h5(h5, images, oneDSetPerImg=False):
        """
        Save images and labels in given HDF5 file.

        Args:
            h5: HDF5 file
            images: the list of images paths
        """
        for i, label_path in enumerate(images):
            # get image path
            img_path = label_path.replace("_dots.png", ".png")
            # get an image as numpy array
            image = np.array(Image.open(img_path), dtype=np.float32) / 255
            image = np.transpose(image, (2, 0, 1))

            # convert a label image into a density map: dataset provides labels
            # in the form on an image with red dots placed in objects position

            # load an RGB image
            label = np.array(Image.open(label_path))
            # make a one-channel label array with 100 in red dots positions
            label = 100.0 * (label[:, :, 0] > 0)
            # generate a density map by applying a Gaussian filter
            if annot_style == 'gaussian':
                label = gaussian_filter(label, sigma=(1, 1), order=0)

            # save data to HDF5 file
            if oneDSetPerImg:
                basename = os.path.basename(label_path).replace("_dots.png", "")
                h5.create_dataset("%s_dots" % basename, (1, 1, *label.shape[0:2]))
                h5.create_dataset(basename, (1, 3, *label.shape[0:2]))
                h5[basename][0] = image
                h5["%s_dots" % basename][0, 0] = label
            else:
                h5["images"][i] = image
                h5["labels"][i, 0] = label

    # use first 150 samples for training and the last 50 for validation
    fill_h5(train_h5, training_image_names, oneDSetPerImg=multiSize)
    fill_h5(valid_h5, validation_image_names, oneDSetPerImg=multiSize)

    # close HDF5 files
    train_h5.close()
    valid_h5.close()


def generate_cell_data():
    """Generate HDF5 files for fluorescent cell dataset."""
    # download and extract dataset
    get_and_unzip(
        "http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip", location="cells"
    )
    # create training and validation HDF5 files
    train_h5, valid_h5 = create_hdf5(
        "cell", train_size=150, valid_size=50, img_size=(256, 256), in_channels=3
    )

    # get the list of all samples
    # dataset name convention: XXXcell.png (image) XXXdots.png (label)
    image_list = glob(os.path.join("cells", "*cell.*"))
    image_list.sort()

    def fill_h5(h5, images):
        """
        Save images and labels in given HDF5 file.

        Args:
            h5: HDF5 file
            images: the list of images paths
        """
        for i, img_path in enumerate(images):
            # get label path
            label_path = img_path.replace("cell.png", "dots.png")
            # get an image as numpy array
            image = np.array(Image.open(img_path), dtype=np.float32) / 255
            image = np.transpose(image, (2, 0, 1))

            # convert a label image into a density map: dataset provides labels
            # in the form on an image with red dots placed in objects position

            # load an RGB image
            label = np.array(Image.open(label_path))
            # make a one-channel label array with 100 in red dots positions
            label = 100.0 * (label[:, :, 0] > 0)
            # generate a density map by applying a Gaussian filter
            label = gaussian_filter(label, sigma=(1, 1), order=0)

            # save data to HDF5 file
            h5["images"][i] = image
            h5["labels"][i, 0] = label

    # use first 150 samples for training and the last 50 for validation
    fill_h5(train_h5, image_list[:150])
    fill_h5(valid_h5, image_list[150:])

    # close HDF5 files
    train_h5.close()
    valid_h5.close()

    # cleanup
    shutil.rmtree("cells")


if __name__ == "__main__":
    get_data()
