import click
import cv2
import h5py
import numpy as np
from skimage.feature import peak_local_max
from PIL import Image
import torch

import base64
from functools import partial
import glob
import json
import os
from pathlib import Path
import platform
import random
import string
import timeit

from data_loader import MultiDimH5Dataset, chooseRandomRegions
from model import UNet, FCRN_A, FCRN_B
from util import *

start_time = timeit.default_timer()


def randID(N=5):
    """Generate uppercase string of alphanumeric characters of length N."""
    return "".join(
        random.SystemRandom().choice(string.ascii_uppercase + string.digits)
        for _ in range(N)
    )


class Predictor:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def predict(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.model_path.split(".")[-1].lower() == "txt":
            with open(self.model_path) as f:
                model_paths = [
                    os.path.join(Path(self.model_path).parent, f"{net_name}.pth")
                    for net_name in f.read().splitlines()
                ]
            for model_path in model_paths:
                self.predict_using_single_model(model_path)
        elif self.glob_models:
            model_paths = glob.glob(self.model_path)
            for model_path in model_paths:
                self.predict_using_single_model(model_path)
        else:
            self.predict_using_single_model(self.model_path)

    def predict_using_single_model(self, model_path):
        true_values = []
        predicted_values = []
        network = {"UNet": UNet, "FCRN_A": FCRN_A, "FCRN_B": FCRN_B}[
            self.network_architecture
        ](
            input_filters=self.input_channels,
            filters=self.unet_filters,
            N=self.convolutions,
        ).to(
            self.device
        )
        network = torch.nn.DataParallel(network)
        network.load_state_dict(torch.load(model_path))
        errors_by_base64str = {}
        print("loaded net:", network)
        data_path = os.path.join(self.dataset_name, "valid.h5")
        # data_path = os.path.join(self.dataset_name, "data.h5")
        dataset = MultiDimH5Dataset(data_path)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            collate_fn=None
            # partial(
            #     chooseRandomRegions,
            #     numPatches=1,
            #     size=160,
            #     #  numPatches=5 if mode == 'train' else 20,
            #     shuffle=True,
            #     skipEmpties=False,
            # ),
        )
        network.train(False)
        counter = 1
        for image, label in dataloader:
            print("Checking image #", counter)
            b64_img_str = str(base64.b64encode(np.ascontiguousarray(image[0].T)))[:100]
            image = image.to(self.device)
            label = label.to(self.device)
            result = network(image)
            for true, predicted in zip(label, result):
                if self.show_imgs:
                    dMapToShow = predicted.cpu().detach().numpy()[0].T
                true_counts = torch.sum(true).item() / 100
                predicted_counts = torch.sum(predicted).item() / 100
                # predicted_counts = len(peak_local_max(dMapToShow, min_distance=2, threshold_abs=0.78))

                # print('what is the sum of the image\'s right border?')
                # print(torch.sum(predicted[:, -3:])/ 100)
                # print('what is the sum of the inner 3/4 along both boundaries?')
                oneEighthImgHt = int(predicted.shape[1] / 8)
                oneEighthImgWidth = int(predicted.shape[2] / 8)
                innerThreeFourths = predicted[
                    :,
                    oneEighthImgHt : predicted.shape[1] - oneEighthImgHt,
                    oneEighthImgWidth : predicted.shape[2] - oneEighthImgWidth,
                ]
                # print(torch.sum(innerThreeFourths)/ 100)
                if self.show_imgs:
                    print("true counts:", true_counts)
                    print("predicted counts:", predicted_counts)
                # print('base64:', b64_img_str)
                true_values.append(true_counts)
                predicted_values.append(predicted_counts)
                # assert b64_img_str not in errors_by_base64str
                errors_by_base64str[b64_img_str] = {
                    "predicted": predicted_counts,
                    "true": true_counts,
                    "abs_error": abs(true_counts - predicted_counts),
                }

                # if abs(true_counts - predicted_counts) > 30:
                # pass
                exampleId = randID()
                if self.show_imgs:
                    imgAsNp = image.cpu().numpy()
                    print("image shape:", image.shape)
                    # print('imgAsNp', imgAsNp)

                    # print("density map shape?", dMapToShow.shape)
                    cv2.imshow("image", cv2.cvtColor(imgAsNp[0].T, cv2.COLOR_BGR2RGB))
                    cv2.imshow("densityMap", dMapToShow)
                    cv2.imshow("ground truth", label[0].cpu().numpy().T)
                    cv2.waitKey(0)

                if self.write_imgs and abs(predicted_counts - true_counts) >= 7:
                    imgAsNp = image.cpu().numpy()
                    dMapToShow = predicted.cpu().detach().numpy()[0].T
                    grayscale_as_color = np.stack((255 * dMapToShow,) * 3, axis=-1)
                    ground_truth_img = 255 * cv2.cvtColor(
                        imgAsNp[0].T, cv2.COLOR_BGR2RGB
                    )
                    # find image with the smaller width.
                    img_list = [
                        grayscale_as_color,
                        ground_truth_img,
                    ]
                    for shape_num in range(2):
                        if (
                            grayscale_as_color.shape[shape_num]
                            != ground_truth_img.shape[shape_num]
                        ):
                            larger_dim = max(img.shape[shape_num] for img in img_list)
                            for i, img in enumerate(img_list):
                                if img.shape[shape_num] < larger_dim:
                                    if shape_num == 0:
                                        new_img = np.zeros(
                                            (larger_dim, img.shape[1], 3)
                                        )
                                    else:
                                        new_img = np.zeros(
                                            (img.shape[0], larger_dim, 3)
                                        )
                                    new_img[0 : img.shape[0], 0 : img.shape[1]] = img
                                    img_list[i] = new_img
                    img_list[1] = Image.fromarray(img_list[1].astype(np.uint8))
                    dMap_pil = Image.fromarray((img_list[0] * 255).astype(np.uint8))
                    dMap_pil.putalpha(dMap_pil.convert("L"))
                    data = np.array(dMap_pil)
                    red, green, blue, alpha = data.T
                    # Replace white with red... (leaves alpha values alone...)
                    white_areas = (red > 0) | (blue > 0) | (green > 0)
                    data[..., :-1][white_areas.T] = (0, 180, 0)  # Transpose back needed
                    dMap_pil = Image.fromarray(data)
                    overlay_image = Image.alpha_composite(
                        img_list[1].convert("RGBA"), dMap_pil
                    )
                    # print("shape of color img:", ground_truth_img.shape)
                    # print("and grayscale:", grayscale_as_color.shape)
                    # if dMapToShow.shape[1] >= dMapToShow.shape[0]:
                    #     stacked_img = np.vstack(tuple(img_list))
                    # else:
                    #     stacked_img = np.hstack(tuple(img_list))

                    # cv2.imwrite(os.path.join('error_examples',
                    # '%s_%i_pred_%i_actual_img.png'%(exampleId, predicted_counts, true_counts)),
                    # 255*cv2.cvtColor(imgAsNp[0].T, cv2.COLOR_BGR2RGB))
                    # cv2.imwrite(os.path.join('error_examples',
                    # '%s_%i_pred_%i_actual_map.png'%(exampleId, predicted_counts, true_counts)),
                    # 255*dMapToShow)
                    Path(f"error_examples_{platform.node()}").mkdir(
                        parents=True, exist_ok=True
                    )
                    cv2.imwrite(
                        os.path.join(
                            f"error_examples_{platform.node()}",
                            "abs_%i_%s_%i_pred_%i_actual_%s.png"
                            % (
                                np.round(abs(predicted_counts - true_counts)).astype(
                                    int
                                ),
                                exampleId,
                                np.round(predicted_counts),
                                np.round(true_counts),
                                os.path.basename(model_path),
                            ),
                        ),
                        # stacked_img,
                        np.array(overlay_image),
                    )
                #     print('predicted: %i\tactual: %i'%(predicted_counts, true_counts))
                #     cv2.imshow('image', cv2.cvtColor(imgAsNp[0].T, cv2.COLOR_BGR2RGB))
                #     cv2.imshow('densityMap', dMapToShow)
                #     cv2.waitKey(0)
                # cv2.imshow('innerSection', innerThreeFourths.cpu().detach().numpy()[0].T)
            counter += 1
        true_values = np.array(true_values)
        # print('true values:', true_values)
        # print('predicted values:', predicted_values)
        abs_diff = np.abs(np.subtract(predicted_values, true_values))
        abs_rel_errors = np.divide(abs_diff, true_values)
        # print('absolute error:', abs_diff)
        # print('relative errors:', abs_rel_errors)
        Path(f"error_results_{platform.node()}").mkdir(parents=True, exist_ok=True)
        with open(
            os.path.join(
                f"error_results_{platform.node()}",
                "error_results_%s.txt" % os.path.basename(model_path),
            ),
            "w",
        ) as myF:
            myF.write(
                "mean absolute error: {number:.{digits}f} ({number})\n".format(
                    number=np.mean(abs_diff), digits=3
                )
            )
            myF.write(
                "mean relative error: {number:.{digits}f} ({number})\n".format(
                    number=np.mean(
                        abs_rel_errors[
                            (abs_rel_errors != np.infty) & (~np.isnan(abs_rel_errors))
                        ]
                    ),
                    digits=3,
                )
            )
            myF.write(
                "mean relative error, 0-10 eggs: {number:.{digits}f} ({number})\n".format(
                    number=np.mean(
                        abs_rel_errors[
                            (abs_rel_errors != np.infty)
                            & (~np.isnan(abs_rel_errors))
                            & (true_values < 11)
                        ]
                    ),
                    digits=3,
                )
            )
            myF.write(
                "mean relative error, 11-40 eggs: {number:.{digits}f} ({number})\n".format(
                    number=np.mean(
                        abs_rel_errors[
                            (abs_rel_errors != np.infty)
                            & ((true_values >= 11) & (true_values < 41))
                        ]
                    ),
                    digits=3,
                )
            )
            myF.write(
                "mean relative error, 41+ eggs: {number:.{digits}f} ({number})\n".format(
                    number=np.mean(
                        abs_rel_errors[
                            (abs_rel_errors != np.infty) & (true_values >= 41)
                        ]
                    ),
                    digits=3,
                )
            )
            myF.write(
                "maximum error: {number:.{digits}f} ({number})\n".format(
                    number=max(abs_diff), digits=3
                )
            )
        with open(
            f"error_results_{platform.node()}/errors_{os.path.basename(model_path)}.json",
            "w",
        ) as myF:
            json.dump(errors_by_base64str, myF, ensure_ascii=False, indent=4)


@click.command()
@click.option(
    "-m",
    "--model_path",
    required=True,
    help="Path of the model to load for measuring accuracy. If a text file instead "
    "of a .pth file, will attempt to load the models listed there, separated by newlines.",
)
@click.option(
    "--glob_models",
    is_flag=True,
    help="Whether to glob the " + "-m/--model_path string to get a list of models.",
)
@click.option(
    "-d",
    "--dataset_name",
    type=click.Choice(
        [
            "cell",
            "egg-eval",
            "mall",
            "ucsd",
            "egg-eval-uli-wt_1",
            "egg-eval-dorsa-wt_1",
            "egg-eval-robert-wt_1",
            "egg-eval-robert-wt_5",
            "egg-eval-robert-task-1",
            "egg-eval-uli-task-1",
            "egg-fullsize-presample-compare-2021-03-22",
            "egg-eval-patch",
            "egg-eval-large-2021-01-26",
            "egg-unshuffled",
        ]
    ),
    required=True,
    help="Dataset by which to evaluate model (expect proper HDF5 files).",
)
@click.option(
    "-n",
    "--network_architecture",
    type=click.Choice(["UNet", "FCRN_A", "FCRN_B"]),
    required=True,
    help="Network architecture to use.",
)
@click.option("--show_imgs", is_flag=True, help="Whether to show images for debugging")
@click.option(
    "--write_imgs",
    is_flag=True,
    help="Whether to write images showing errors of 7 eggs or greater",
)
@click.option("--dropout", is_flag=True, help="Whether the net uses dropout")
@click.option(
    "-inCh",
    "--input_channels",
    default=3,
    help="Number of input channels in evaluation images.",
)
@click.option(
    "--unet_filters",
    default=64,
    help="Number of filters for U-Net convolutional layers.",
)
@click.option(
    "--convolutions", default=2, help="Number of layers in a convolutional block."
)
def eval_model(
    model_path,
    glob_models,
    dataset_name,
    network_architecture,
    show_imgs,
    write_imgs,
    dropout,
    input_channels,
    unet_filters,
    convolutions,
):
    Predictor(
        model_path=model_path,
        glob_models=glob_models,
        dataset_name=dataset_name,
        network_architecture=network_architecture,
        show_imgs=show_imgs,
        write_imgs=write_imgs,
        dropout=dropout,
        input_channels=input_channels,
        unet_filters=unet_filters,
        convolutions=convolutions,
    ).predict()
    print("Total run time:", timeit.default_timer() - start_time)


if __name__ == "__main__":
    eval_model()
