"""
Author: Jyothi Vishnu Vardhan Kolla
CS-7180 Fall 2023

This file contains the implementations for SRCNN, SRGAN
"""

import torch.nn as nn

# This class implements the SRCNN model.


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # No padding to avoid border effects.
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=1, padding=2)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=3, kernel_size=5, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
