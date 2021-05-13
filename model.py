"""The implementation of U-Net and FCRN-A models."""
from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models import resnet

from model_config import DROPOUT_PROB

from unet_parts import *

import timeit


class UOut(nn.Module):
    """Add random noise to every layer of the net."""

    def forward(self, input_tensor: torch.Tensor):
        if not self.training:
            return input_tensor
        with torch.cuda.device(0):
            return (
                input_tensor
                + 2
                * DROPOUT_PROB
                * torch.cuda.FloatTensor(input_tensor.shape).uniform_()
                - DROPOUT_PROB
            )


class ResNet(nn.Module):
    def __init__(self, module, in_channels, out_channels, stride):
        super().__init__()
        self.module = module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, inputs):
        output = self.module(inputs)
        skip = None
        if self.stride != 1 or self.in_channels != self.out_channels:
            skip = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=self.stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
            )
        identity = inputs
        if skip is not None:
            skip = skip.cuda()
            identity = skip(inputs)
        output += identity
        return output


class SamePadder(nn.Module):
    def __init__(self, filter_shape):
        super(SamePadder, self).__init__()
        self.filter_shape = filter_shape

    def forward(self, input):
        strides = (None, 1, 1)
        in_height, in_width = input.shape[2:4]
        filter_height, filter_width = self.filter_shape

        if in_height % strides[1] == 0:
            pad_along_height = max(filter_height - strides[1], 0)
        else:
            pad_along_height = max(filter_height - (in_height % strides[1]), 0)
        if in_width % strides[2] == 0:
            pad_along_width = max(filter_width - strides[2], 0)
        else:
            pad_along_width = max(filter_width - (in_width % strides[2]), 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom))


def pad_to_nearest_even(input):
    in_height, in_width = input.shape[1:3]
    if in_height % 2 == 1:
        pad_top = 1
    else:
        pad_top = 0
    if in_width % 2 == 1:
        pad_left = 1
    else:
        pad_left = 0
    return F.pad(input, (pad_left, 0, pad_top, 0))


class BlockBuilder:
    """Create convolutional blocks for building neural nets."""

    def conv_block(
        self,
        channels: Tuple[int, int],
        size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        N: int = 1,
    ):
        """
        Create a block with N convolutional layers with ReLU activation function.
        The first layer is IN x OUT, and all others - OUT x OUT.

        Args:
            channels: (IN, OUT) - no. of input and output channels
            size: kernel size (fixed for all convolution in a block)
            stride: stride (fixed for all convolution in a block)
            N: no. of convolutional layers

        Returns:
            A sequential container of N convolutional layers.
        """
        # a single convolution + batch normalization + ReLU block
        def block(in_channels):
            layers = [
                SamePadder(size),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=channels[1],
                    kernel_size=size,
                    stride=stride,
                    bias=False,
                    # padding=(size[0] // 2, size[1] // 2),
                ),
                nn.BatchNorm2d(num_features=channels[1]),
                nn.ReLU(),
            ]
            return nn.Sequential(*layers)

        # create and return a sequential container of convolutional layers
        # input size = channels[0] for first block and channels[1] for all others
        return nn.Sequential(*[block(channels[bool(i)]) for i in range(N)])


class ConvCat(nn.Module):
    """Convolution with upsampling + concatenate block."""

    def __init__(
        self,
        channels: Tuple[int, int],
        size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        N: int = 1,
    ):
        """
        Create a sequential container with convolutional block (see conv_block)
        with N convolutional layers and upsampling by factor 2.
        """
        super(ConvCat, self).__init__()
        bb = BlockBuilder()
        self.conv = nn.Sequential(
            bb.conv_block(channels, size, stride, N), nn.Upsample(scale_factor=2)
        )

    def forward(self, to_conv: torch.Tensor, to_cat: torch.Tensor):
        """Forward pass.

        Args:
            to_conv: input passed to convolutional block and upsampling
            to_cat: input concatenated with the output of a conv block
        """
        return torch.cat([self.conv(to_conv), to_cat], dim=1)


class FCRN_B(nn.Module):
    """
    Fully Convolutional Regression Network B

    Ref. W. Xie et al. 'Microscopy Cell Counting with Fully Convolutional
    Regression Networks'
    """

    def __init__(self, N: int = 1, input_filters: int = 3, **kwargs):
        """
        Create FCRN-B model with:

            * kernel size = (3, 3) for downsampling and (5, 5) for upsampling
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2

        Args:
            N: no. of convolutional layers per block (see conv_block)
            input_filters: no. of input channels
        """
        super(FCRN_B, self).__init__()
        bb = BlockBuilder()
        self.model = nn.Sequential(
            # original version of FCRN-B
            # # downsampling
            # bb.conv_block(channels=(input_filters, 32), size=(3, 3), N=N),
            # bb.conv_block(channels=(32, 64), size=(3, 3), N=N),
            # nn.MaxPool2d(2),
            # bb.conv_block(channels=(64, 128), size=(3, 3), N=N),
            # # "convolutional fully connected"
            # bb.conv_block(channels=(128, 256), size=(3, 3), N=N),
            # nn.MaxPool2d(2),
            # # upsampling
            # bb.conv_block(channels=(256, 256), size=(5, 5), N=N),
            # nn.Upsample(scale_factor=2),
            # bb.conv_block(channels=(256, 256), size=(5, 5), N=N),
            # nn.Upsample(scale_factor=2),
            # bb.conv_block(channels=(256, 1), size=(5, 5), N=N),
            # FCRN-B, but with FCRN-A's channel configurations
            # in the upsampling layers.
            # downsampling
            bb.conv_block(channels=(input_filters, 32), size=(3, 3), N=N),
            nn.MaxPool2d(2),
            bb.conv_block(channels=(32, 64), size=(3, 3), N=N),
            bb.conv_block(channels=(64, 128), size=(3, 3), N=N),
            nn.MaxPool2d(2),
            # "convolutional fully connected"
            bb.conv_block(channels=(128, 256), size=(3, 3), N=N),
            # upsampling
            bb.conv_block(channels=(256, 128), size=(5, 5), N=N),
            nn.Upsample(scale_factor=2),
            bb.conv_block(channels=(128, 64), size=(5, 5), N=N),
            nn.Upsample(scale_factor=2),
            bb.conv_block(channels=(64, 1), size=(5, 5), N=N),
        )

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        return self.model(input)


class FCRN_A(nn.Module):
    """
    Fully Convolutional Regression Network A

    Ref. W. Xie et al. 'Microscopy Cell Counting with Fully Convolutional
    Regression Networks'
    """

    def __init__(self, N: int = 1, input_filters: int = 3, **kwargs):
        """
        Create FCRN-A model with:

            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * no. of filters as defined in an original model:
              input size -> 32 -> 64 -> 128 -> 512 -> 128 -> 64 -> 1

        Args:
            N: no. of convolutional layers per block (see conv_block)
            input_filters: no. of input channels
        """
        super(FCRN_A, self).__init__()
        self.input_filters = input_filters
        self.N = N
        bb = BlockBuilder()
        self.model = nn.Sequential(
            # downsampling
            bb.conv_block(channels=(input_filters, 32), size=(3, 3), N=N),
            nn.MaxPool2d(2),
            bb.conv_block(channels=(32, 64), size=(3, 3), N=N),
            nn.MaxPool2d(2),
            bb.conv_block(channels=(64, 128), size=(3, 3), N=N),
            nn.MaxPool2d(2),
            # "convolutional fully connected"
            bb.conv_block(channels=(128, 512), size=(3, 3), N=N),
            # upsampling
            nn.Upsample(scale_factor=2),
            bb.conv_block(channels=(512, 128), size=(3, 3), N=N),
            nn.Upsample(scale_factor=2),
            bb.conv_block(channels=(128, 64), size=(3, 3), N=N),
            nn.Upsample(scale_factor=2),
            bb.conv_block(channels=(64, 1), size=(3, 3), N=N),
        )
        # self.block1 = nn.Sequential(
        #     *[bb.conv_block(channels=(input_filters, 32), size=(3, 3), N=N)]
        # )
        # self.block2 = nn.Sequential(
        #     *[bb.conv_block(channels=(32, 64), size=(3, 3), N=N)]
        # )
        # self.block3 = nn.Sequential(
        #     *[bb.conv_block(channels=(64, 128), size=(3, 3), N=N)]
        # )
        # self.block4 = nn.Sequential(
        #     *[bb.conv_block(channels=(128, 512), size=(3, 3), N=N)]
        # )
        # self.block5 = nn.Sequential(
        #     *[bb.conv_block(channels=(512, 128), size=(3, 3), N=N)]
        # )
        # self.block6 = nn.Sequential(
        #     *[bb.conv_block(channels=(128, 64), size=(3, 3), N=N)]
        # )
        # self.block7 = nn.Sequential(
        #     *[bb.conv_block(channels=(64, 1), size=(3, 3), N=N)]
        # )

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        return self.model(input)
        print("shape before:", input.shape)
        pool = nn.MaxPool2d(2)
        upsample = nn.Upsample(scale_factor=2)
        input = self.block1(input)
        # input = pad_to_nearest_even(input)
        print("shape after block 1:", input.shape)
        input = pool(input)
        print("shape after max pool:", input.shape)
        input = self.block2(input)
        # input = pad_to_nearest_even(input)
        print("shape after block2:", input.shape)
        input = pool(input)
        print("shape after max pool:", input.shape)
        input = self.block3(input)
        # input = pad_to_nearest_even(input)
        print("shape after:", input.shape)
        input = pool(input)
        print("shape after max pool:", input.shape)
        input = self.block4(input)
        # input = pad_to_nearest_even(input)
        print("shape after block4:", input.shape)
        input = upsample(input)
        print("shape after upsampling:", input.shape)
        input = self.block5(input)
        print("shape after block5:", input.shape)
        input = upsample(input)
        print("shape after upsampling:", input.shape)
        input = self.block6(input)
        print("shape after block6:", input.shape)
        input = upsample(input)
        print("shape after upsampling:", input.shape)
        input = self.block7(input)
        print("shape after block7 (end):", input.shape)


class UNet(nn.Module):
    def __init__(self, input_filters, filters=None, N=None, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = input_filters
        # self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(input_filters, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)
        self.density_pred = nn.Conv2d(
            in_channels=filters, out_channels=1, kernel_size=(1, 1), bias=False
        )

    def forward(self, x):
        # start_t = timeit.default_timer()
        x1 = self.inc(x)
        # cp_1 = timeit.default_timer()
        # print('time for first convolutions:', cp_1 - start_t)
        x2 = self.down1(x1)
        # cp_2 = timeit.default_timer()
        # print('time for 2nd convolutions:', cp_2 - cp_1)
        x3 = self.down2(x2)
        # cp_3 = timeit.default_timer()
        # print('time for 3rd convolutions:', cp_3 - cp_2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # logits = self.outc(x)
        return self.density_pred(x)
        # return logits


class UNet_old(nn.Module):
    """
    U-Net implementation.

    Ref. O. Ronneberger et al. "U-net: Convolutional networks for biomedical
    image segmentation."
    """

    def __init__(self, filters: int = 64, input_filters: int = 3, **kwargs):
        """
        Create U-Net model with:

            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * fixed no. of convolutional layers per block = 2 (see conv_block)
            * constant no. of filters for convolutional layers

        Args:
            filters: no. of filters for convolutional layers
            input_filters: no. of input channels
        """
        super(UNet, self).__init__()
        # first block channels size
        initial_filters = (input_filters, filters)
        # channels size for downsampling
        down_filters = (filters, filters)
        # channels size for upsampling (input doubled because of concatenate)
        up_filters = (2 * filters, filters)
        bb = BlockBuilder()

        # downsampling
        self.block1 = bb.conv_block(channels=initial_filters, size=(3, 3), N=2)
        self.block2 = bb.conv_block(channels=down_filters, size=(3, 3), N=2)
        self.block3 = bb.conv_block(channels=down_filters, size=(3, 3), N=2)

        # upsampling
        self.block4 = ConvCat(channels=down_filters, size=(3, 3), N=2)
        self.block5 = ConvCat(channels=up_filters, size=(3, 3), N=2)
        self.block6 = ConvCat(channels=up_filters, size=(3, 3), N=2)

        # density prediction
        self.block7 = bb.conv_block(channels=up_filters, size=(3, 3), N=2)
        self.density_pred = nn.Conv2d(
            in_channels=filters, out_channels=1, kernel_size=(1, 1), bias=False
        )

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        # use the same max pooling kernel size (2, 2) across the network
        pool = nn.MaxPool2d(2)

        # downsampling
        block1 = self.block1(input)
        pool1 = pool(block1)
        block2 = self.block2(pool1)
        pool2 = pool(block2)
        block3 = self.block3(pool2)
        pool3 = pool(block3)

        # upsampling
        block4 = self.block4(pool3, block3)
        block5 = self.block5(block4, block2)
        block6 = self.block6(block5, block1)

        # density prediction
        block7 = self.block7(block6)
        return self.density_pred(block7)


# --- PYTESTS --- #


def run_network(network: nn.Module, input_channels: int):
    """Generate a random image, run through network, and check output size."""
    sample = torch.ones((1, input_channels, 224, 224))
    result = network(input_filters=input_channels)(sample)
    assert result.shape == (1, 1, 224, 224)


def test_UNet_color():
    """Test U-Net on RGB images."""
    run_network(UNet, 3)


def test_UNet_grayscale():
    """Test U-Net on grayscale images."""
    run_network(UNet, 1)


def test_FRCN_color():
    """Test FCRN-A on RGB images."""
    run_network(FCRN_A, 3)


def test_FRCN_grayscale():
    """Test FCRN-A on grayscale images."""
    run_network(FCRN_A, 1)
