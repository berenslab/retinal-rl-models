from collections import OrderedDict

import torch

from retinal_rl_models.base_model import BaseModel
from retinal_rl_models.util import assert_list


class FullyConnected(BaseModel):
    def __init__(
        self,
        inp_size: int = 1024,
        out_size: int = 10,
        num_layers: int = 3,
        hidden_dims: int = 1024,
        act_name="elu",
    ):
        super().__init__(locals())

        self.hidden_dims = assert_list(hidden_dims, self.num_layers - 1)

        fcs = []

        for i in range(num_layers):
            _in_size = self.inp_size if i == 0 else self.hidden_dims[i - 1]
            _out_size = self.out_size if i == num_layers - 1 else self.hidden_dims[i]
            fcs.append(
                (
                    "fc" + str(i),
                    torch.nn.Linear(_in_size, _out_size),
                )
            )
            if i < num_layers - 1:
                fcs.append(
                    (
                        self.act_name + str(i),
                        self.str_to_activation(self.act_name),
                    )
                )
        self.fcs = torch.nn.Sequential(OrderedDict(fcs))

    def forward(self, x):
        x = self.fcs(x)
        if not self.training:
            x = torch.nn.functional.softmax(x)
        return x
