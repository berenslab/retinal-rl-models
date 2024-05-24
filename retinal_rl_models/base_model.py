import os
from abc import ABC

import torch
import torch.nn as nn
import yaml


class BaseModel(nn.Module, ABC):
    def __init__(self, init_params: dict) -> None:
        """
        Initializes the base model.
        All params in the dictionary will be added as instance parameters.

        init_params: the parameters used to instantiate a model. Simplest way to pass them on: call locals()
        """
        super().__init__()

        # Store all parameters as config
        self._config = init_params
        self._config.pop("self")
        self._config.pop("__class__")
        self.__dict__.update(self._config)

    @property
    def config(self) -> dict:
        conf = self._config
        return {"type": self.__class__, "config": conf}

    def save(self, filename, save_cfg=True):
        config = self.config
        if save_cfg:
            with open(filename + ".cfg", "w") as f:
                yaml.dump(config, f)
        torch.save(self.state_dict(), filename + ".pth")

    @staticmethod
    def load(model_path, weights_file=None):
        with open(model_path + ".cfg", "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        model = config["type"](**config["config"])

        if weights_file is None:
            weights_file = model_path + ".pth"
        if os.path.exists(weights_file):
            try:
                model.load_state_dict(torch.load(weights_file))
            except:
                model.load_state_dict(
                    torch.load(weights_file, map_location=torch.device("cpu"))
                )
        return model

    @staticmethod
    def str_to_activation(act: str) -> nn.Module:
        act = str.lower(act)
        if act == "elu":
            return nn.ELU(inplace=True)
        elif act == "relu":
            return nn.ReLU(inplace=True)
        elif act == "tanh":
            return nn.Tanh()
        elif act == "softplus":
            return nn.Softplus()
        elif act == "identity":
            return nn.Identity(inplace=True)
        else:
            raise Exception("Unknown activation function")

    @staticmethod
    def calc_num_elements(module: nn.Module, module_input_shape: tuple[int]):
        shape_with_batch_dim = (1,) + module_input_shape
        some_input = torch.rand(shape_with_batch_dim)
        num_elements = module(some_input).numel()
        return num_elements
