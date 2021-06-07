"""Looper implementation."""
from os import error
from typing import Optional, List

import cv2
from dual_loss_helper import get_loss_weight_function
import torch
import numpy as np
import matplotlib
import matplotlib.axes
import timeit


class Looper:
    """Looper handles epoch loops, logging, and plotting."""

    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
        config: dict,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader,
        dataset_size: int,
        plots: Optional[matplotlib.axes.Axes] = None,
        validation: bool = False,
        left_col_plots: str = None,
        rand_samp_mult: int = None,
    ):
        """
        Initialize Looper.

        Args:
            network: already initialized model
            device: a device model is working on
            config: dict of settings for the training
            optimizer: already initialized optimizer link to network parameters
            data_loader: already initialized data loader
            dataset_size: no. of samples in dataset
            plot: matplotlib axes
            validation: flag to set train or eval mode

        """
        self.network = network
        self.device = device
        self.loss_tp = config["loss"]
        self.config = config
        self.should_quit_early = False
        self.set_up_loss()
        self.optimizer = optimizer
        self.loader = data_loader
        self.validation = validation
        self.plots = plots
        self.running_loss = []
        if self.loss_tp == "dual":
            self.running_loss_by_pixel = []
            self.running_loss_by_egg_ct = []
        self.running_mean_abs_err = []
        self.mean_abs_err_ylim_max = 2
        if plots is not None:
            assert (
                left_col_plots is not None
            ), "left_col_plots must have a value if plots are set"
        self.left_col_plots = left_col_plots
        self.rand_samp_mult = rand_samp_mult
        self.size = dataset_size * (
            rand_samp_mult if rand_samp_mult is not None and not validation else 1
        )

    def set_up_loss(self):
        def mean_abs_error_loss(result, label):
            errors = []
            for true, predicted in zip(label, result):
                actual = torch.sum(true)
                errors.append(
                    torch.abs(actual - torch.sum(predicted))
                    / 100
                    / self.config["maeLossDivisor"]
                )
            return torch.mean(torch.stack(errors))

        def mse_false_pos_penalty_loss(result, label):
            diffs = result - label
            coeffs = torch.where(diffs > 0, 1., 1.)
            scaled_diffs = torch.mul(diffs, coeffs)
            return (scaled_diffs**2).mean()

        if self.loss_tp == "mse":
            self.loss = torch.nn.MSELoss()
        elif self.loss_tp == "mae":
            self.loss = mean_abs_error_loss
        elif self.loss_tp == "dual":
            self.coeffs = []
            self.n_batches_since_reset = 0
            # self.loss_mse = torch.nn.MSELoss()
            self.loss_mse = mse_false_pos_penalty_loss
            self.loss_weight_fn = get_loss_weight_function(self.config)

            def dual_loss(result, label):
                if (
                    self.n_batches_since_reset >= self.config["dualOptions"]["period"]
                    or len(self.coeffs) == 0
                ):
                    self.coeffs = self.loss_weight_fn(self.epoch_number)
                    self.n_batches_since_reset = 0
                self.n_batches_since_reset += 1
                loss_1 = self.loss_mse(result, label)
                loss_2 = mean_abs_error_loss(result, label)
                loss_sum = (self.coeffs[0] * loss_1).add(loss_2 * self.coeffs[1])
                return loss_sum, loss_1, loss_2

            self.loss = dual_loss

    def run(self, epoch_number: int):
        """Run a single epoch loop.

        Returns:
            Mean absolute error.
        """
        # reset current results and add next entry for running loss
        self.true_values = []
        self.predicted_values = []
        self.err = []
        self.abs_err = []
        self.mean_err = []
        self.running_loss.append(0)
        self.epoch_number = epoch_number
        if self.loss_tp == "dual":
            self.running_loss_by_pixel.append(0)
            self.running_loss_by_egg_ct.append(0)

        # set a proper mode: train or eval
        self.network.train(not self.validation)

        for image, label in self.loader:
            # abs_err_for_batch = []
            # move images and labels to given device
            image = image.to(self.device)
            label = label.to(self.device)

            # clear accumulated gradient if in train mode
            if not self.validation:
                self.optimizer.zero_grad()

            # get model prediction (a density map)
            result = self.network(image)
            asymmetryCorrs = {"v": 0, "h": 0}
            hDiff = label.shape[-1] - result.shape[-1]
            if hDiff % 2 > 0:
                asymmetryCorrs["h"] = 1
            else:
                asymmetryCorrs["h"] = 0
            vDiff = label.shape[-2] - result.shape[-2]
            # vertExcess = int(vDiff / 2)
            if vDiff % 2 > 0:
                asymmetryCorrs["v"] = 1
            else:
                asymmetryCorrs["v"] = 0

            # loop over batch samples
            for true, predicted in zip(label, result):
                # integrate a density map to get no. of objects
                # note: density maps were normalized to 100 * no. of objects
                #       to make network learn better
                true_counts = torch.sum(true).item() / 100
                predicted_counts = torch.sum(predicted).item() / 100
                # print("true counts:", true_counts)
                # print("and predicted:", predicted_counts)

                # update current epoch results
                self.err.append(true_counts - predicted_counts)
                self.abs_err.append(abs(self.err[-1]))
                # abs_err_for_batch.append(abs(self.err[-1]))
                self.true_values.append(true_counts)
                self.predicted_values.append(predicted_counts)
            # self.abs_err += abs_err_for_batch
            # print("absolute errors used for the loss calculation:", abs_err_for_batch)
            # input()

            label = label[:, :, : label.shape[-2] - vDiff, : label.shape[-1] - hDiff]

            # calculate loss and update running loss
            if self.loss_tp != "dual":
                loss = self.loss(result, label)
            else:
                loss_results = self.loss(result, label)
                loss = loss_results[0]
                # not sure how to plot this.
                self.running_loss_by_pixel[-1] += (
                    image.shape[0] * loss_results[1].item() / self.size
                )
                self.running_loss_by_egg_ct[-1] += (
                    image.shape[0] * loss_results[2].item() / self.size
                )

            self.running_loss[-1] += image.shape[0] * loss.item() / self.size

            # update weights if in train mode
            if not self.validation:
                loss.backward()
                self.optimizer.step()

        # calculate errors and standard deviation
        self.update_errors()

        if (
            not self.validation
            and self.config["abandonDivergentTraining"]
            and self.epoch_number + 1 == self.config["minNumEpochs"]
            and not np.any(
                np.asarray(self.running_mean_abs_err) <= self.mean_abs_err_ylim_max
            )
        ):
            self.should_quit_early = True
            return

        # update live plot
        if self.plots is not None:
            self.plot()

        # print epoch summary
        self.log()
        # print("how many steps in the epoch?", counter)

        return self.mean_abs_err

    def update_errors(self):
        """
        Calculate errors and standard deviation based on current
        true and predicted values.
        """

        # self.abs_err = [abs(error) for error in self.err]
        self.mean_err = sum(self.err) / self.size
        self.mean_abs_err = sum(self.abs_err) / self.size
        self.running_mean_abs_err.append(self.mean_abs_err)
        self.std = np.array(self.err).std()

    def plot(self):
        """Plot true vs predicted counts and loss."""
        # true vs predicted counts
        true_line = [[0, max(self.true_values)]] * 2  # y = x
        epochs = np.arange(1, len(self.running_loss) + 1)
        self.plots[0].cla()
        self.plots[0].set_title("Train" if not self.validation else "Valid")

        if self.left_col_plots == "scatter":
            self.plots[0].set_xlabel("True value")
            self.plots[0].set_ylabel("Predicted value")
            self.plots[0].plot(*true_line, "r-")
            self.plots[0].scatter(self.true_values, self.predicted_values)
        elif self.left_col_plots == "mae":
            self.plots[0].set_xlabel("Epoch")
            self.plots[0].set_ylabel("Mean absolute error")
            self.plots[0].set_ylim((0, self.mean_abs_err_ylim_max))
            self.plots[0].plot(epochs, self.running_mean_abs_err)

        # loss
        self.plots[1].cla()
        self.plots[1].set_title("Train" if not self.validation else "Valid")
        self.plots[1].set_xlabel("Epoch")
        self.plots[1].set_ylabel("Loss")
        self.plots[1].set_ylim((0, 0.2))
        self.plots[1].plot(
            epochs,
            self.running_loss,
            "h-"
            if self.loss_tp == "dual" and self.config["dualOptions"] == "flip"
            else "-",
            label="Loss" if self.loss_tp == "dual" else None,
            markersize=9,
        )
        if self.loss_tp == "dual":
            self.plots[1].plot(
                epochs, self.running_loss_by_egg_ct, label="Egg count loss"
            )
            self.plots[1].plot(
                epochs, self.running_loss_by_pixel, label="Pixel-wise loss"
            )
            self.plots[1].legend()

        matplotlib.pyplot.pause(0.01)
        matplotlib.pyplot.tight_layout()

    def log(self):
        """Print current epoch results."""
        print(
            f"{'Train' if not self.validation else 'Valid'}:\n"
            f"\tAverage loss: {self.running_loss[-1]:3.4f}\n"
            f"\tMean error: {self.mean_err:3.3f}\n"
            f"\tMean absolute error: {self.mean_abs_err:3.3f}\n"
            f"\tError deviation: {self.std:3.3f}"
        )
