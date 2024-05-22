import torch
from torch import nn
from enum import Enum

class Activation(Enum):
    elu = nn.ELU(inplace=True),
    relu = nn.ReLU(inplace=True),
    tanh = nn.Tanh(),
    softplus = nn.Softplus(),
    identity = nn.Identity()


def activation(act) -> nn.Module:
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
    
def calc_num_elements(module, module_input_shape):
    shape_with_batch_dim = (1,) + module_input_shape
    some_input = torch.rand(shape_with_batch_dim)
    num_elements = module(some_input).numel()
    return num_elements

def assert_list(list_candidate, len_list, dtype = int):
        if isinstance(list_candidate, dtype):
            _list = [list_candidate] * len_list
        else:
            assert len(list_candidate) == len_list
            _list = list_candidate
        return _list