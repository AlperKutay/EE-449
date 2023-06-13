import os

import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import utils
import numpy as np


def my_conv2d(input, kernel):
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape  # I wrote "_" since we had input_channels value from above

    out_height = in_height - kernel_height + 1
    out_width = in_width - kernel_width + 1

    # initialize output tensor
    out = torch.zeros(batch_size, out_channels, out_height, out_width)

    for b in range(batch_size):
        for c_out in range(out_channels):
            for i in range(out_height):
                for j in range(out_width):
                    # extract the input patch corresponding to the current output pixel
                    patch = input[b, :, i: i + kernel_height, j: j + kernel_width]

                    # apply the kernel to the patch and sum the result
                    out[b, c_out, i, j] = torch.sum(torch.from_numpy(patch) * kernel[c_out])

    return out


# input shape: [batch size, input_channels, input_height, input_width]
input = np.load("data\\samples_7.npy")
# input shape: [output_channels, input_channels, filter_height, filter width]
kernel = np.load("data\\kernel.npy")
out = my_conv2d(input, kernel)
utils.part2Plots(out, 64, "ResultQ2\\", "my_first_conv2D")
