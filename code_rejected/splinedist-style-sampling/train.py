"""Main script used to train networks."""
import datetime
import os
import platform
import signal
import timeit
from typing import Union, Optional, List

import click
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot

from data_loader import H5Dataset, MultiDimH5Dataset, padImagesBasedOnMaxSize, chooseRandomRegions
from looper import Looper
from model import UNet, FCRN_A, FCRN_B
from randomSampler import RandomSampler


@click.command()
@click.option('-d', '--dataset_name',
              type=click.Choice(['cell', 'egg', 'egg-combined', 'egg-fullsize',
              'egg-patch-0', 'mall', 'ucsd']), required=True,
              help='Dataset to train model on (expect proper HDF5 files).')
@click.option('-n', '--network_architecture',
              type=click.Choice(['UNet', 'FCRN_A', 'FCRN_B']),
              required=True,
              help='Model to train.')
@click.option('-m', '--model_path', default='', help='Path of an existing ' +
              'model to load')
@click.option('-lr', '--learning_rate', default=1e-2,
              help='Initial learning rate (lr_scheduler is applied).')
@click.option('-e', '--epochs', default=150, help='Number of training epochs.')
@click.option('--batch_size', default=6,
              help='Batch size for both training and validation dataloaders.')
@click.option('-hf', '--horizontal_flip', default=0.0,
              help='The probability of horizontal flip for training dataset.')
@click.option('-vf', '--vertical_flip', default=0.0,
              help='The probability of vertical flip for training dataset.')
@click.option('-rot', '--rand_rotate', is_flag=True,
              help='Randomly rotate images in training dataset.')
@click.option('--rand_brightness', is_flag=True, help='Randomly vary ' +
              'brightness of images in training dataset.')
# @click.option('--dropout', is_flag=True, help='Activate dropout')
@click.option('--unet_filters', default=64,
              help='Number of filters for U-Net convolutional layers.')
@click.option('--convolutions', default=2,
              help='Number of layers in a convolutional block.')
@click.option('--plot', is_flag=True, help="Generate a live plot.")
@click.option('--export_at_end', is_flag=True, help='Save the model and the ' +\
    'plot from the final epoch of training only, with the timestamp of the ' +\
    "training's end in their filenames.")
def train(dataset_name: str,
          network_architecture: str,
          model_path: str,
          learning_rate: float,
          epochs: int,
          batch_size: int,
          horizontal_flip: float,
          vertical_flip: float,
          rand_rotate: bool,
          rand_brightness: bool,
        #   dropout: bool,
          unet_filters: int,
          convolutions: int,
          plot: bool,
          export_at_end: bool):
    """Train chosen model on selected dataset."""
    # use GPU if avilable
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = {}     # training and validation HDF5-based datasets
    dataloader = {}  # training and validation dataloaders

    for mode in ['train', 'valid']:
        # expected HDF5 files in dataset_name/(train | valid).h5
        data_path = os.path.join(dataset_name, f"{mode}.h5")
        # turn on flips only for training dataset
        if dataset_name in ('egg-fullsize', 'egg-combined'):
            h5Constructor = MultiDimH5Dataset
            # batch_size = 1
        else:
            h5Constructor = H5Dataset
        dataset[mode] = h5Constructor(data_path,
                                  horizontal_flip if mode == 'train' else 0,
                                  vertical_flip if mode == 'train' else 0,
                                  rand_rotate if mode == 'train' else False,
                                  rand_brightness if mode == 'train' else False)
        dataloader[mode] = torch.utils.data.DataLoader(dataset[mode],
                                                       batch_size=batch_size if mode == 'train' else 1,
                                                       sampler=RandomSampler(
                                                           dataset[mode],
                                                           num_samples=400) if mode == 'train' else None,
                                                       collate_fn=chooseRandomRegions
                                                       )

    # only UCSD dataset provides greyscale images instead of RGB
    input_channels = 1 if dataset_name == 'ucsd' else 3

    # initialize a model based on chosen network_architecture
    network = {
        'UNet': UNet,
        'FCRN_A': FCRN_A,
        'FCRN_B': FCRN_B
    }[network_architecture](input_filters=input_channels,
                            filters=unet_filters,
                            N=convolutions,
                            ).to(device)
    network = torch.nn.DataParallel(network)
    if model_path != '':
        network.load_state_dict(torch.load(model_path))

    # initialize loss, optimized and learning rate scheduler
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=1e-5)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=5,
    #                                                gamma=0.95)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.25, verbose=True)

    # if plot flag is on, create a live plot (to be updated by Looper)
    if plot:
        pyplot.ion()
        fig, plots = pyplot.subplots(nrows=2, ncols=2)
    else:
        plots = [None] * 2

    # create training and validation Loopers to handle a single epoch
    train_looper = Looper(network, device, loss, optimizer,
                          dataloader['train'], len(dataset['train']), plots[0],)
    valid_looper = Looper(network, device, loss, optimizer,
                          dataloader['valid'], len(dataset['valid']), plots[1],
                          validation=True)

    # current best results (lowest mean absolute error on validation set)
    current_best = np.infty

    for i, epoch in enumerate((range(epochs))):
        start_time = timeit.default_timer()
        print(f"Epoch {epoch + 1}\n")

        # run training epoch and update learning rate
        train_looper.run()

        # run validation epoch
        with torch.no_grad():
            result = valid_looper.run()

        lr_scheduler.step(result)
        # update checkpoint if new best is reached
        newBest = result < current_best
        reachedSaveInterval = i%20 == 0
        if newBest:
            current_best = result
        if not export_at_end and (newBest or reachedSaveInterval):
            if newBest:
                torch.save(network.state_dict(),
                       f'{dataset_name}_{network_architecture}_best.pth')
                print(f"\nNew best result: {result}")
            if reachedSaveInterval:
                torch.save(network.state_dict(),
                       f'{dataset_name}_{network_architecture}_iter{i}.pth')
                print('Saving a regular interval export')

        print('single-epoch duration:', timeit.default_timer() - start_time)
        print("\n", "-"*80, "\n", sep='')

    if export_at_end:
        torch.save(network.state_dict(),
            f'{dataset_name}_{network_architecture}_{epochs}epochs_' + \
            f'{platform.node()}_{datetime.datetime.now()}.pth'.replace(
            ':', '-'))
        pyplot.savefig(f'{dataset_name}_{network_architecture}_{epochs}epochs_' + \
            f'{platform.node()}_{datetime.datetime.now()}.png'.replace(
            ':', '-'))
    print(f"[Training done] Best result: {current_best}")
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    train()
