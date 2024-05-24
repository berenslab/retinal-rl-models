# retinal-rl-models
Code for the pytorch models used in the retinal-rl project (and related non-rl usages)

## Installation

After cloning the repository, go into the toplevel directory and run:
```
pip install -e .
```

## Usage

To load a model config, execute the following code:
```
from retinal_rl_models.base_model import BaseModel

model = BaseModel.load("configs/stride_downsample.cfg")
```
The function automatically imports the correct class based on the config. If there is a .pth file with the same name, the weights from that file are also loaded.