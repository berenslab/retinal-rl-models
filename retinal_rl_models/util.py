from torch import nn

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