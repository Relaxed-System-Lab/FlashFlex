import torch.nn as nn 
from typing import Tuple

class PipeSequential(nn.Sequential):
    """
    Pipe variant of ``nn.Sequential`` which supports multiple inputs.
    """

    def forward(self, *inputs):
        for module in self:
            if isinstance(inputs, Tuple):  # type: ignore[arg-type]
                inputs = module(*inputs)
            else:
                # Don't expand single variables (ex: lists/Tensor)
                inputs = module(inputs)
        return inputs