"""Looper implementation."""
from os import error
from typing import Optional, List

import cv2
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
        loss_tp: str,
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
            loss: the cost function
            optimizer: already initialized optimizer link to network parameters
            data_loader: already initialized data loader
            dataset_size: no. of samples in dataset
            plot: matplotlib axes
            validation: flag to set train or eval mode

        """
        self.network = network
        self.device = device
        self.loss_tp = loss_tp
        self.set_up_loss()
        self.optimizer = optimizer
        self.loader = data_loader
        self.validation = validation
        self.plots = plots
        self.running_loss = []
        self.running_mean_abs_err = []
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
        if self.loss_tp == "mse":
            self.loss = torch.nn.MSELoss()
        elif self.loss_tp == "mae":

            # def mean_abs_error_loss(abs_errors):
            #     return torch.mean(abs_errors)

            def mean_abs_error_loss(result, label):
                errors = []
                for true, predicted in zip(label, result):
                    actual = torch.sum(true)
                    errors.append(
                        torch.abs(actual - torch.sum(predicted)) / 100 / 4 / 5
                    )
                return torch.mean(torch.stack(errors))

            self.loss = mean_abs_error_loss

    def run(self, i=None):
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
            if self.loss_tp == "mse":
                loss = self.loss(result, label)
            elif self.loss_tp == "mae":
                # abs_err_for_batch = torch.tensor(abs_err_for_batch, requires_grad=True)
                loss = self.loss(result, label)
            # print("what is the loss function result?", loss)
            self.running_loss[-1] += image.shape[0] * loss.item() / self.size

            # update weights if in train mode
            if not self.validation:
                weights_pre_optim = list(self.network.parameters())[0].clone()
                loss.backward()
                self.optimizer.step()
                weights_post_optim = list(self.network.parameters())[0].clone()
                # print(
                #     "are weights the same after optimization?",
                #     torch.equal(weights_pre_optim, weights_post_optim),
                # )

        # calculate errors and standard deviation
        self.update_errors()

        # update live plot
        if self.plots is not None:
            self.plot()

        # print epoch summary
        self.log()
        # print("how many steps in the epoch?", counter)

        return self.mean_abs_err
        # if the loss increases by a magnitude of about 100, it seems reasonable to me that the learning rate might have to decrease by a similar amount for the predictions to come out similarly. Ultimately, this proved to be the case, because I started seeing good results after having decreased learning rate to the level 1.25e-8.

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
            self.plots[0].set_ylim((0, 2))
            self.plots[0].plot(epochs, self.running_mean_abs_err)

        # loss
        self.plots[1].cla()
        self.plots[1].set_title("Train" if not self.validation else "Valid")
        self.plots[1].set_xlabel("Epoch")
        self.plots[1].set_ylabel("Loss")
        self.plots[1].set_ylim((0, 120))
        self.plots[1].plot(epochs, self.running_loss)

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
