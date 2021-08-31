# Copyright 2021 Sebastian Wichmann
# Licensed under the Apache License, Version 2.0 (the "License")


import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):
    def __init__(
            self,
            input_dim=23,
            output_dim=512,
            context_size=5,
            stride=1,
            dilation=1,
            batch_norm=True,
            dropout_p=0.0):
        super(TDNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_size = context_size
        self.stride = stride
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm


class ConvTDNN(TDNN):
    def __init__(self,
                 input_dim=23,
                 output_dim=512,
                 context_size=5,
                 stride=1,
                 dilation=1,
                 batch_norm=True,
                 dropout_p=0.0):
        super(ConvTDNN, self).__init__(input_dim, output_dim, context_size, stride, dilation, batch_norm, dropout_p)

        module_list = nn.ModuleList()
        module_list.append(nn.Conv1d(input_dim, output_dim, kernel_size=context_size, stride=stride, dilation=dilation))
        module_list.append(nn.ReLU())

        if self.batch_norm:
            module_list.append = nn.BatchNorm1d(output_dim)
        if dropout_p:
            module_list.append(nn.Dropout(p=dropout_p))

        self.layers = nn.Sequential(*module_list)

    def forward(self, x):
        _, _, d = x.shape
        assert (d == self.input_dim), f'Input dimension was wrong. Expected ({self.input_dim}), got ({d})'

        return self.layers(x)


class LinearTDNN(TDNN):
    # This class matches the descrption mentioned in paper
    #   https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
    # All credit for this class to https://github.com/cvqluu/TDNN

    def __init__(self,
                 input_dim=23,
                 output_dim=512,
                 context_size=5,
                 stride=1,
                 dilation=1,
                 batch_norm=True,
                 dropout_p=0.0):
        super(LinearTDNN, self).__init__(input_dim, output_dim, context_size, stride, dilation, batch_norm, dropout_p)

        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        self.non_linearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        _, _, d = x.shape
        assert (d == self.input_dim), f'Input dimension was wrong. Expected ({self.input_dim}), got ({d})'
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.context_size, self.input_dim),
            stride=(1, self.input_dim),
            dilation=(self.dilation, 1)
        )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x)
        x = self.non_linearity(x)

        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        return x
