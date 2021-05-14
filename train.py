"""Main script used to train networks."""
import cv2
from glob import glob
import h5py
from randomSampler import RandomSampler
from model import UNet, FCRN_A, FCRN_B
from looper import Looper
from data_loader import (
    DatasetWithAugs,
    H5Dataset,
    MultiDimH5Dataset,
    chooseRandomRegions,
)
from matplotlib import pyplot
import datetime
from functools import partial
import json
import os
from pathlib import Path
import platform
from random import shuffle
import signal
import time
import timeit

import click
import torch
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

DEFAULT_CONFIG_PATH = "./configs/default.json"

# temp command: python trainByBatchLinux.py "-d egg-fullsize-pt-presample-compare-2021-03-23 -n FCRN_A -lr 0.00025 -e 300 -hf 0.5 -vf 0.5 --val_interval 2 -rot --plot --batch_size 4 --rand_samp_mult 20 --config /media/Synology3/Robert/objects_counting_dmap/configs/dualLossWithZoom_2021-05-13.json" --n_repeats 10

def get_dataloader(
    dataset,
    mode,
    config,
    patch_size,
    rand_samp_mult,
    batch_size,
    custom_collate=True,
    apply_gaussian=False,
    debug=False,
):
    if custom_collate:
        coll_fn = partial(
            chooseRandomRegions,
            numPatches=1,
            size=patch_size,
            #  numPatches=5 if mode == 'train' else 20,
            shuffle=True if mode == "train" else False,
            skipEmpties=config["skipEmpties"],
            randomPatchSizes=config["randomPatchSizes"],
            zoomMin=config["zoomMin"],
            zoomMax=config["zoomMax"],
            applyGaussian=apply_gaussian,
            debug=debug,
        )
    else:
        coll_fn = None
    if mode != "train" or rand_samp_mult is None:
        sampler = None
    elif rand_samp_mult is not None and mode == "train":
        sampler = (
            RandomSampler(
                dataset[mode], num_samples=rand_samp_mult * len(dataset[mode].images)
            )
            if mode == "train"
            else None
        )
    return torch.utils.data.DataLoader(
        dataset[mode],
        batch_size=batch_size if mode == "train" else 1,
        sampler=sampler,
        collate_fn=coll_fn,
    )


def splitAndSampleDatasets(
    config,
    dataset_name,
    patch_size,
    rand_samp_mult,
    batch_size,
    horizontal_flip,
    vertical_flip,
    rand_brightness,
    rand_rotate,
    dataset,
    dataloader,
):
    if "train_proportion" in config:
        train_proportion = config["train_proportion"]
    else:
        train_proportion = 0.79
    all_images = h5py.File(os.path.join(dataset_name, "data.h5"), "r")
    image_to_mask = {
        key: "%s_dots" % key
        for key in set([keyName.replace("_dots", "") for keyName in all_images.keys()])
    }
    image_names = list(image_to_mask.keys())
    shuffle(image_names)
    split_index = round(len(image_names) * train_proportion)
    train_imgs = image_names[:split_index]
    val_imgs = image_names[split_index:]
    print("\n# train imgs:", len(train_imgs))
    print("# val imgs:", len(val_imgs))
    sampled_patches = []

    for i, val_img in enumerate(val_imgs):
        img = all_images[val_img][0]
        mask = all_images[image_to_mask[val_img]][0]
        result = chooseRandomRegions(
            [[img, mask]],
            20,
            patch_size,
            skipEmpties=config["skipEmpties"],
            randomPatchSizes=config["randomPatchSizes"],
            zoomMin=config["zoomMin"],
            zoomMax=config["zoomMax"],
            debug=False,
        )
        for i in range(result[0].shape[0]):
            sampled_patches.append([result[0][i].numpy(), result[1][i].numpy()])
    temp_validation_dataset = DatasetWithAugs(
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rand_rotate=False,
        rand_brightness=rand_brightness,
    )
    for patch in sampled_patches:
        temp_validation_dataset.images.append(patch[0])
        temp_validation_dataset.labels.append(patch[1])
    dataset["valid"] = DatasetWithAugs()
    for item in temp_validation_dataset:
        dataset["valid"].images.append(item[0])
        dataset["valid"].labels.append(item[1])
    dataset["train"] = DatasetWithAugs(
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rand_rotate=rand_rotate,
        rand_brightness=rand_brightness,
    )
    # fill the training dataset.
    for trn_img in train_imgs:
        dataset["train"].images.append(all_images[trn_img][0])
        dataset["train"].labels.append(all_images[image_to_mask[trn_img]][0])
    for mode in ("train", "valid"):
        dataloader[mode] = get_dataloader(
            dataset,
            mode,
            config,
            patch_size,
            rand_samp_mult,
            batch_size,
            custom_collate=mode == "train",
        )


class FileNameHelper:
    def __init__(
        self, dataset_name, network_architecture, train_ts, existing_model=None
    ) -> None:
        self.dataset_name = dataset_name
        self.network_architecture = network_architecture
        self.train_ts = train_ts
        self.existing_model = existing_model
        if self.existing_model is not None:
            self.existing_model = os.path.basename(self.existing_model).split(".pth")[0]

    def retrain_descriptor(self):
        if self.existing_model is None or self.existing_model == '':
            return ""

        return f"_retrain_{self.existing_model}_"

    def existing_weights_regex(self):
        return f"{self.name_and_arch()}_best*.pth"

    def name_and_arch(self, use_current_time=False):
        time_str = datetime.datetime.now() if use_current_time else self.train_ts
        return (
            f"{self.dataset_name}{self.retrain_descriptor()}"
            f"_{self.network_architecture}_{platform.node()}_{time_str}"
        ).replace(":", "-")


@click.command()
@click.option(
    "-d",
    "--dataset_name",
    type=click.Choice(
        [
            "cell",
            "egg",
            "egg-combined",
            "egg-fullsize",
            "egg-fullsize-presample-compare-2021-03-22",
            "egg-fullsize-pt-presample-compare-2021-03-23",
            "egg-patch-0",
            "egg-patch-presample-compare-2021-03-22",
            "egg-unshuffled",
            "egg-unshuffled-archive-2021-01-04",
            "mall",
            "ucsd",
        ]
    ),
    required=True,
    help="Dataset to train model on (expect proper HDF5 files).",
)
@click.option(
    "-n",
    "--network_architecture",
    type=click.Choice(["UNet", "FCRN_A", "FCRN_B"]),
    required=True,
    help="Model to train.",
)
@click.option(
    "-m", "--model_path", default="", help="Path of an existing model to load"
)
@click.option(
    "--name_map",
    default="",
    help="Path to a JSON file whose keys include the basename of the net "
    "passed in through -m and whose values are set to the original names of the nets."
    " This is used if net names are shortened as part of a paired experiment.",
)
@click.option(
    "--resume_lr",
    is_flag=True,
    help="If starting a training from an existing model, then use the learning rate"
    " from the end of that preceding training. Overrides value passed in through -lr."
    " Learning rate history files should be in the same folder as the existing model"
    " (-m)",
)
@click.option(
    "-lr",
    "--learning_rate",
    default=1e-2,
    help="Initial learning rate (lr_scheduler is applied).",
)
@click.option(
    "--lr_scheduler",
    type=click.Choice(["plateau", "step"]),
    default="plateau",
    help="Type of learning rate scheduler to use:" ' "plateau" (default) or "step"',
)
@click.option("-e", "--epochs", default=150, help="Number of training epochs.")
@click.option(
    "--val_interval",
    default=20,
    help="Frequency, in epochs, at which to measure validation performance",
)
@click.option(
    "--batch_size",
    default=6,
    help="Batch size for both training and validation dataloaders.",
)
@click.option(
    "--patch_size",
    default=160,
    help="Side length in pixels of the patches sampled from images",
)
@click.option(
    "--rand_samp_mult",
    default=None,
    type=int,
    help="How many times to include each full-size training image in a single"
    "epoch (by default, each image gets included only once per epoch)",
)
@click.option(
    "-hf",
    "--horizontal_flip",
    default=0.0,
    help="The probability of horizontal flip for training dataset.",
)
@click.option(
    "-vf",
    "--vertical_flip",
    default=0.0,
    help="The probability of vertical flip for training dataset.",
)
@click.option(
    "-rot",
    "--rand_rotate",
    is_flag=True,
    help="Randomly rotate images in training dataset.",
)
@click.option(
    "--rand_brightness",
    is_flag=True,
    help="Randomly vary " + "brightness of images in training dataset.",
)
@click.option(
    "--unet_filters",
    default=64,
    help="Number of filters for U-Net convolutional layers.",
)
@click.option(
    "--convolutions", default=2, help="Number of layers in a convolutional block."
)
@click.option("--plot", is_flag=True, help="Generate a live plot.")
@click.option(
    "--left_col_plots",
    type=click.Choice(["mae", "scatter"]),
    default="mae",
    help="Plot type to be displayed in the left column"
    ": mae (mean absolute error) or scatter (scatter plot of"
    " individual predictions from the latest epoch)",
)
@click.option(
    "--export_at_end",
    is_flag=True,
    help="Save the model and the "
    + "plot from the final epoch of training only, with the timestamp of the "
    + "training's end in their filenames.",
)
@click.option(
    "--config",
    help="Path to a file with config settings for the"
    + " training. Note: any settings from command line will override these.",
    default=DEFAULT_CONFIG_PATH,
)
def train(
    dataset_name: str,
    network_architecture: str,
    model_path: str,
    name_map: str,
    resume_lr: bool,
    learning_rate: float,
    lr_scheduler: str,
    epochs: int,
    val_interval: int,
    batch_size: int,
    patch_size: int,
    rand_samp_mult: int,
    horizontal_flip: float,
    vertical_flip: float,
    rand_rotate: bool,
    rand_brightness: bool,
    unet_filters: int,
    convolutions: int,
    plot: bool,
    left_col_plots: str,
    export_at_end: bool,
    config: str,
):
    """Train chosen model on selected dataset."""
    # use GPU if avilable
    train_ts = datetime.datetime.now()
    with open(DEFAULT_CONFIG_PATH) as myF:
        config_default = json.load(myF)
    if not config:
        config = DEFAULT_CONFIG_PATH
    with open(config) as myF:
        config = json.load(myF)
    for key in (
        "skipEmpties",
        "randomPatchSizes",
        "zoomMin",
        "zoomMax",
        "sampleMode",
        "loss",
    ):
        if not key in config:
            config[key] = config_default[key]
    if name_map:
        with open(name_map, "r") as f:
            name_map = json.load(f)
    fname_helper = FileNameHelper(
        dataset_name, network_architecture, train_ts, model_path
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = {}  # training and validation HDF5-based datasets
    dataloader = {}  # training and validation dataloaders

    lr_history = {}
    lr_history_filename = f"{fname_helper.name_and_arch()}_lr_history.json"

    def write_lr_history():
        with open(lr_history_filename, "w") as my_f:
            json.dump(lr_history, my_f, ensure_ascii=False, indent=4)

    def load_train_val_sets():
        for mode in ("train", "valid"):
            data_path = os.path.join(dataset_name, f"{mode}.h5")
            if dataset_name in ("egg-fullsize", "egg-combined"):
                h5Constructor = MultiDimH5Dataset
            else:
                h5Constructor = H5Dataset
            dataset[mode] = h5Constructor(
                data_path,
                horizontal_flip if mode == "train" else 0,
                vertical_flip if mode == "train" else 0,
                rand_rotate if mode == "train" else False,
                rand_brightness if mode == "train" else False,
                # debug=mode == 'train'
            )
            dataloader[mode] = get_dataloader(
                dataset,
                mode,
                config,
                patch_size,
                rand_samp_mult,
                batch_size,
                custom_collate=False,
                debug=False,
            )

    if "egg-unshuffled" in dataset_name or config["sampleMode"] == "dynamic":
        already_split = os.path.isfile(os.path.join(dataset_name, "train.h5"))
        if not already_split:
            splitAndSampleDatasets(
                config,
                dataset_name,
                patch_size,
                rand_samp_mult,
                batch_size,
                horizontal_flip,
                vertical_flip,
                rand_brightness,
                rand_rotate,
                dataset,
                dataloader,
            )
        elif already_split and config["sampleMode"] == "dynamic":
            validation_samples = []
            validation_fullsize = h5py.File(os.path.join(dataset_name, "valid.h5"), "r")
            for k in validation_fullsize:
                if "_dots" in k:
                    continue
                patches = chooseRandomRegions(
                    [
                        [
                            validation_fullsize[k][0],
                            validation_fullsize["%s_dots" % k][0],
                        ]
                    ],
                    20,
                    patch_size,
                    skipEmpties=config["skipEmpties"],
                    randomPatchSizes=config["randomPatchSizes"],
                    zoomMin=config["zoomMin"],
                    zoomMax=config["zoomMax"],
                    applyGaussian=True,
                    debug=False,
                )
                for i in range(patches[0].shape[0]):
                    validation_samples.append(
                        [patches[0][i].numpy(), patches[1][i].numpy()]
                    )
            temp_validation_dataset = DatasetWithAugs(
                horizontal_flip=horizontal_flip,
                vertical_flip=vertical_flip,
                rand_rotate=rand_rotate,
                rand_brightness=rand_brightness,
            )
            for sample in validation_samples:
                temp_validation_dataset.images.append(sample[0])
                temp_validation_dataset.labels.append(sample[1])
            dataset["valid"] = DatasetWithAugs()
            for item in temp_validation_dataset:
                dataset["valid"].images.append(item[0])
                dataset["valid"].labels.append(item[1])
            dataset["train"] = DatasetWithAugs(
                horizontal_flip=horizontal_flip,
                vertical_flip=vertical_flip,
                rand_rotate=rand_rotate,
                rand_brightness=rand_brightness,
            )
            train_fullsize = h5py.File(os.path.join(dataset_name, "train.h5"), "r")
            for k in train_fullsize:
                if "_dots" in k:
                    continue
                dataset["train"].images.append(train_fullsize[k][0])
                dataset["train"].labels.append(train_fullsize["%s_dots" % k][0])
            for mode in ("train", "valid"):
                dataloader[mode] = get_dataloader(
                    dataset,
                    mode,
                    config,
                    patch_size,
                    rand_samp_mult,
                    batch_size,
                    custom_collate=mode == "train",
                    apply_gaussian=True,
                    debug=False,
                )
    else:
        load_train_val_sets()

    # only UCSD dataset provides greyscale images instead of RGB
    input_channels = 1 if dataset_name == "ucsd" else 3

    # initialize a model based on chosen network_architecture
    network = {"UNet": UNet, "FCRN_A": FCRN_A, "FCRN_B": FCRN_B}[network_architecture](
        input_filters=input_channels,
        filters=unet_filters,
        N=convolutions,
    ).to(device)
    network = torch.nn.DataParallel(network)
    if model_path != "":
        network.load_state_dict(torch.load(model_path))

    # initialize loss, optimized and learning rate scheduler
    # loss = torch.nn.MSELoss()
    if resume_lr:
        parent_dir = Path(model_path).parents[0]
        if name_map:
            search_string = "_".join(
                name_map[os.path.basename(model_path).split(".pth")[0]].split("_")[-2:]
            ).split(".pth")[0]
        else:
            search_string = "_".join(model_path.split("_")[-2:]).split(".pth")[0]
        lr_files = glob(
            os.path.join(parent_dir, "*lr_history*%s*" % search_string)
        ) + glob(os.path.join(parent_dir, "*%s*lr_history*" % search_string))
        if len(lr_files) > 0:
            with open(lr_files[0]) as f:
                lrs = json.load(f)
            learning_rate = list(lrs.values())[-1]
    optimizer = torch.optim.SGD(
        network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5
    )
    lr_scheduler_opt = lr_scheduler
    if lr_scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=75, gamma=0.5
        )
    elif lr_scheduler == "plateau":
        if "rlrParams" in config:
            print("setting learning rate from config file")
            rlrFactor = config["rlrParams"]["factor"]
            patience = config["rlrParams"]["patience"]
        else:
            print("using default learning rate settings")
            rlrFactor = 0.25
            patience = 10
        print("Learning rate factor:", rlrFactor, "\tLearning rate patience:", patience)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=rlrFactor, verbose=True, patience=patience
        )

    # if plot flag is on, create a live plot (to be updated by Looper)
    if plot:
        pyplot.ion()
        fig, plots = pyplot.subplots(nrows=2, ncols=2)
    else:
        plots = [None] * 2

    # create training and validation Loopers to handle a single epoch
    train_looper = Looper(
        network,
        device,
        config["loss"],
        optimizer,
        dataloader["train"],
        len(dataset["train"]),
        plots[0],
        left_col_plots=left_col_plots,
        rand_samp_mult=rand_samp_mult,
    )
    valid_looper = Looper(
        network,
        device,
        config["loss"],
        optimizer,
        dataloader["valid"],
        len(dataset["valid"]),
        plots[1],
        validation=True,
        left_col_plots=left_col_plots,
    )

    # current best results (lowest mean absolute error on validation set)
    current_best = np.infty

    for i, epoch in enumerate((range(epochs))):
        current_lr = optimizer.param_groups[0]["lr"]
        if len(lr_history) == 0 or lr_history["last"] != current_lr:
            lr_history["last"] = current_lr
            lr_history[f"epoch_{i+1}"] = current_lr
            write_lr_history()
        start_time = timeit.default_timer()
        print(f"Epoch {epoch + 1}\n")

        # run training epoch and update learning rate
        train_looper.run(i)

        if i % val_interval == 0:
            # run validation epoch
            with torch.no_grad():
                result = valid_looper.run()

            if lr_scheduler_opt == 'plateau':
                lr_scheduler.step(result)
            else:
                lr_scheduler.step()
            # update checkpoint if new best is reached
            newBest = result < current_best
            reachedSaveInterval = i % 20 == 0
            if newBest:
                current_best = result
                for f in glob(fname_helper.existing_weights_regex()):
                    os.unlink(f)
                torch.save(
                    network.state_dict(),
                    f"{fname_helper.name_and_arch()}_best_epoch{i + 1}.pth",
                )
                print(f"\nNew best result: {result}")
            if not export_at_end and reachedSaveInterval:
                torch.save(
                    network.state_dict(), f"{fname_helper.name_and_arch()}_iter{i}.pth"
                )
                print("Saving a regular interval export")

        print("single-epoch duration:", timeit.default_timer() - start_time)
        print("\n", "-" * 80, "\n", sep="")

    if export_at_end:
        torch.save(
            network.state_dict(),
            f"{fname_helper.name_and_arch()}_{epochs}epochs.pth",
        )
        pyplot.savefig(f"{fname_helper.name_and_arch()}_{epochs}epochs.png")
    print(f"[Training done] Best result: {current_best}")
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    train()
