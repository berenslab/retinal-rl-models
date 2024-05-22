import torch.nn as nn
import torch
from torch.nn.modules.utils import _pair
from retinal_rl_models.util import calc_num_elements, assert_list


class DownsampleCNN(nn.Module):
    def __init__(
        self,
        inp_shape: tuple[int, int],
        out_size: int,
        num_layers: int = 3,
        fc_dim: int = 128,
        fc_in_size: int = None,
        in_channels: int = 3,
        num_channels: int | list[int] = 16,
        kernel_size: int | list[int] = 3,
        stride: int | list[int] = 1,
        padding: int | list[int] = 0,
        dilation: int | list[int] = 1,
        activation="relu",
        pooling_ks: int | list[int] = 1,
    ):
        self.config = locals()  # Store all parameters as config

        # add parameters to model
        self.inp_shape = inp_shape
        self.act = activation
        self.num_layers = num_layers
        self.out_size = out_size
        self.fc_dim = fc_dim
        self.fc_in_size = fc_in_size
        self.in_channels = in_channels
        self.padding = assert_list(padding, self.num_layers)
        self.dilation = assert_list(dilation, self.num_layers)
        self.num_channels = assert_list(num_channels, self.num_layers)
        self.kernel_size = assert_list(kernel_size, self.num_layers)
        self.stride = assert_list(stride, self.num_layers)
        self.pooling_ks = assert_list(pooling_ks, self.num_layers)

        self.pool = [
            nn.AvgPool2d(kernel_size=self.pooling_ks[i]) for i in range(self.num_layers)
        ]
        if self.fc_in_size != None:
            self.pool[-1] = nn.AdaptiveAvgPool2d(output_size=fc_in_size)

        # Define the first convolutional layer
        self.conv1 = torch.nn.Conv2d(
            self.in_channels,
            self.num_channels[0],
            self.kernel_size[0],
            self.stride[0],
            self.padding[i + 1],
            self.dilation[i + 1],
        )

        # Define additional convolutional layers if needed
        self.extra_conv_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.extra_conv_layers.append(
                torch.nn.Conv2d(
                    self.num_channels[i],
                    self.num_channels[i + 1],
                    self.kernel_size[i + 1],
                    self.stride[i + 1],
                    self.padding[i + 1],
                    self.dilation[i + 1],
                )
            )

        # Fully connected layer
        fc_in = calc_num_elements(self.conv_head, self.inp_shape)
        self.fc = nn.Linear(fc_in, self.out_size)

    def forward_conv(self, x):
        if self.spd != 1:
            x = self.space_to_depth(x)
        x = self.conv1(x)
        x = self._activation_func(x)

        for pool_ks, pool, conv_layer in zip(
            self.pooling_ks[:-1], self.pool[:-1], self.extra_conv_layers
        ):
            if pool_ks != 1:
                x = pool(x)
            if self.spd != 1:
                x = self.space_to_depth(x)
            x = conv_layer(x)
            x = self._activation_func(x)
        if self.fc_in_size != None:
            x = self.pool[-1](x)
        elif self.pooling_ks[-1] != 1:
            x = self.pool[-1](x)
        return x

    def forward(self, x):
        x = self.forward_conv(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if not self.training:
            x = self.softmax(x)
        return x
