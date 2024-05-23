# retinal-rl-models
Code for the pytorch models used in the retinal-rl project (and related non-rl usages)

## Installation

After cloning the repository, go into the toplevel directory and run:
'''
pip install -e .
'''

## Usage

To load a model config, execute the following code:
'''
from retinal_rl_models.encoder.vision_models import GenericModel

model = GenericModel.load("retinal_rl_models/configs/stride_downsampl.cfg")
'''
