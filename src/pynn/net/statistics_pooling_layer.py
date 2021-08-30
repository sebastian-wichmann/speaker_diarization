# Copyright 2021 Sebastian Wichmann
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
import torch.nn as nn


class StatisticsPoolingLayer(nn.Module):

    def __init__(self, input_dim, calc_dim: int = 1):
        super(StatisticsPoolingLayer, self).__init__()

        self.input_dim = input_dim
        self.dim = calc_dim

    def forward(self, x):
        assert len(x.shape) == 3, "Input vector must be 3D but is {}D".format(len(x.shape))
        _, _, d = x.shape
        assert d == self.input_dim, "{} must be {}".format(d, self.input_dim)

        return torch.cat((
            torch.mean(x, self.dim),
            torch.std(x, self.dim)),
            self.dim
        )
