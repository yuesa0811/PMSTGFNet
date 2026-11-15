# GTS Feature Fusion

This repository provides the core implementation of the Gated Temporal-Spatial (GTS) Feature Fusion module used in PMSTGFNet. The module adaptively recalibrates temporal and spatial features, constructs an interaction representation, and applies channel shuffle to interleave the three feature groups.

## File

- `GTS_Feature_Fusion.py`: PyTorch implementation of the GTS Feature Fusion module.

## Requirements

- Python 3.8 or later
- PyTorch

## Usage

```python
import torch

from GTS_Feature_Fusion import GTS_Feature_Fusion

channels = 60
module = GTS_Feature_Fusion(channels)

x_time = torch.randn(8, channels, 22, 128)
x_space = torch.randn(8, channels, 22, 128)
output = module(x_time, x_space)

print(output.shape)  # torch.Size([8, 180, 22, 128])
```

The two inputs must have the same shape `(batch_size, channels, height, width)`. The output contains `3 * channels` channels.

## Availability

The complete PMSTGFNet model code will be released after the paper is accepted.
