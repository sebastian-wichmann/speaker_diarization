# Copyright 2021 Sebastian Wichmann
# Licensed under the Apache License, Version 2.0 (the "License")

import torch.nn as nn

class TDNN(nn.Module):

    def __init__(self,
                 input_dim=23,
                 output_dim=512,
                 context_size=5,
                 stride=1,
                 dilation=1,
                 dropout=0.0):
        super(TDNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_size = context_size
        self.stride = stride
        self.dilation = dilation
        self.dropout = dropout

        module_list = nn.ModuleList()
        module_list.append(nn.Conv1d(input_dim, output_dim, kernel_size=context_size, stride=stride, dilation=dilation))
        module_list.append(nn.ReLU())

        if dropout:
            module_list.append(nn.Dropout(p=dropout))

        self.layers = nn.Sequential(*module_list)

    def forward(self, x):
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)

        return self.layers(x)
