def conv_block(self, channels: Tuple[int, int],
                   size: Tuple[int, int],
                   stride: Tuple[int, int] = (1, 1),
                   N: int = 1):
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
                nn.Conv2d(in_channels=in_channels,
                          out_channels=channels[1],
                          kernel_size=size,
                          stride=stride,
                          bias=False,
                          padding=(size[0] // 2, size[1] // 2)),
                nn.BatchNorm2d(num_features=channels[1]),
                nn.ReLU()
            ]
            # if self.dropout:
            #     layers.append(UOut())
            # layers.append(nn.BatchNorm2d(num_features=channels[1]))
            # layers = [ResNet(nn.Sequential(nn.Conv2d(in_channels=in_channels,
            #                                          out_channels=channels[1],
            #                                          kernel_size=size,
            #                                          stride=stride,
            #                                          bias=False,
            #                                          padding=(size[0] // 2, size[1] // 2))), in_channels, channels[1], stride),
            #           nn.ReLU()]
            # if self.dropout:
                # layers.append(UOut())
            # postActivation = nn.Sequential(*layers)
            # return nn.Sequential(postActivation, nn.BatchNorm2d(num_features=channels[1]))
            return nn.Sequential(*layers)
        # create and return a sequential container of convolutional layers
        # input size = channels[0] for first block and channels[1] for all others
        return nn.Sequential(*[block(channels[bool(i)]) for i in range(N)])