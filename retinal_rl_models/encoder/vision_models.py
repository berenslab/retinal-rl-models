import torch
from torch import nn
from retinal_rl_models.util import activation
from collections import OrderedDict

class RetinalModel(Encoder):

    def __init__(self, cfg: Config, obs_space: ObsSpace):

        super().__init__(cfg)

        # Activation function
        self.act_name = cfg.activation
        self.nl_fc = activation(cfg.activation)

        # Saving parameters
        self.bp_chans = cfg.base_channels
        self.rgc_chans = self.bp_chans * 2
        self.v1_chans = self.rgc_chans * 2

        if cfg.retinal_bottleneck is not None:
            self.btl_chans = cfg.retinal_bottleneck
        else:
            self.btl_chans = self.rgc_chans

        # Pooling
        self.spool = 3
        self.mpool = 4

        # Padding
        self.spad = 0  # padder(self.spool)
        self.mpad = 0  # padder(self.mpool)

        # Preparing Conv Layers
        conv_layers = OrderedDict(
            [
                (
                    "bp_filters",
                    nn.Conv2d(3, self.bp_chans, self.spool, padding=self.spad),
                ),
                ("bp_outputs", activation(self.act_name)),
                ("bp_averages", nn.AvgPool2d(self.spool, ceil_mode=True)),
                (
                    "rgc_filters",
                    nn.Conv2d(
                        self.bp_chans, self.rgc_chans, self.spool, padding=self.spad
                    ),
                ),
                ("rgc_outputs", activation(self.act_name)),
                ("rgc_averages", nn.AvgPool2d(self.spool, ceil_mode=True)),
                ("btl_filters", nn.Conv2d(self.rgc_chans, self.btl_chans, 1)),
                ("btl_outputs", activation(self.act_name)),
                (
                    "v1_filters",
                    nn.Conv2d(
                        self.btl_chans, self.v1_chans, self.mpool, padding=self.mpad
                    ),
                ),
                ("v1_simple_outputs", activation(self.act_name)),
                ("v1_complex_outputs", nn.MaxPool2d(self.mpool, ceil_mode=True)),
            ]
        )

        self.conv_head = nn.Sequential(conv_layers)

        cout_hght, cout_wdth = encoder_out_size(self.conv_head, *obs_space.shape[1:])
        self.conv_head_out_size = cout_hght * cout_wdth * self.v1_chans
        self.encoder_out_size = cfg.rnn_size
        self.fc1 = nn.Linear(self.conv_head_out_size, self.encoder_out_size)

    def forward(self, x):

        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.nl_fc(self.fc1(x))
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


# Retinal Stride Encoder


class RetinalStrideModel(Encoder):

    def __init__(self, cfg: Config, obs_space: ObsSpace):

        super().__init__(cfg)

        # Activation function
        self.act_name = cfg.activation
        self.nl_fc = activation(cfg.activation)

        # Saving parameters
        self.bp_chans = cfg.base_channels
        self.rgc_chans = self.bp_chans * 2
        self.v1_chans = self.rgc_chans * 2

        if cfg.retinal_bottleneck is not None:
            self.btl_chans = cfg.retinal_bottleneck
        else:
            self.btl_chans = self.rgc_chans

        # Pooling
        self.spool = 3
        self.mpool = 4

        # Padding
        self.spad = 0  # padder(self.spool)
        self.mpad = 0  # padder(self.mpool)

        # Preparing Conv Layers
        conv_layers = OrderedDict(
            [
                (
                    "bp_filters",
                    nn.Conv2d(
                        3,
                        self.bp_chans,
                        self.spool,
                        stride=self.spool,
                        padding=self.spad,
                    ),
                ),
                ("bp_outputs", activation(self.act_name)),
                (
                    "rgc_filters",
                    nn.Conv2d(
                        self.bp_chans, self.rgc_chans, self.spool, padding=self.spad
                    ),
                ),
                ("rgc_outputs", activation(self.act_name)),
                ("rgc_averages", nn.AvgPool2d(self.spool, ceil_mode=True)),
                ("btl_filters", nn.Conv2d(self.rgc_chans, self.btl_chans, 1)),
                ("btl_outputs", activation(self.act_name)),
                (
                    "v1_filters",
                    nn.Conv2d(
                        self.btl_chans, self.v1_chans, self.mpool, padding=self.mpad
                    ),
                ),
                ("v1_simple_outputs", activation(self.act_name)),
                ("v1_complex_outputs", nn.MaxPool2d(self.mpool, ceil_mode=True)),
            ]
        )

        self.conv_head = nn.Sequential(conv_layers)

        cout_hght, cout_wdth = encoder_out_size(self.conv_head, *obs_space.shape[1:])
        self.conv_head_out_size = cout_hght * cout_wdth * self.v1_chans
        self.encoder_out_size = cfg.rnn_size
        self.fc1 = nn.Linear(self.conv_head_out_size, self.encoder_out_size)

    def forward(self, x):

        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.nl_fc(self.fc1(x))
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


# Prototypical Encoder


class PrototypicalModel(Encoder):

    def __init__(self, cfg: Config, obs_space: ObsSpace):

        super().__init__(cfg)

        self.nl_fc = activation(cfg.activation)
        self.act_name = cfg.activation
        # self.krnsz = cfg.kernel_size
        # self.gchans = cfg.base_channels
        # self.pad = (self.krnsz - 1) // 2

        # Preparing Conv Layers
        # [[input_channels, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
        conv_layers = OrderedDict(
            [
                ("conv1_filters", nn.Conv2d(3, 32, 8, stride=4)),
                ("conv1_output", activation(self.act_name)),
                ("conv2_filters", nn.Conv2d(32, 64, 4, stride=2)),
                ("conv2_output", activation(self.act_name)),
                ("conv3_filters", nn.Conv2d(64, 128, 3, stride=2)),
                ("conv3_output", activation(self.act_name)),
            ]
        )

        self.conv_head = nn.Sequential(conv_layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space.shape)
        self.encoder_out_size = cfg.rnn_size
        self.fc1 = nn.Linear(self.conv_head_out_size, self.encoder_out_size)

    def forward(self, x):

        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.nl_fc(self.fc1(x))
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size
