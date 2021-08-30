# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import math

import torch
import torch.nn as nn

from . import XavierLinear
from .s2s_transformer import Encoder

class Transformer(nn.Module):
    def __init__(self, n_classes, d_input, d_model, d_inner, n_layer, n_head, d_project=0,
            dropout=0.1, layer_drop=0., time_ds=1, use_cnn=False):
        super().__init__()

        d_inner = d_model*4 if d_inner == 0 else d_inner
        self.encoder = Encoder(d_input, d_model, n_layer, n_head, d_inner, use_cnn=use_cnn, 
                               time_ds=time_ds, dropout=dropout, layer_drop=layer_drop)
        d_project = d_model if d_project==0 else d_project
        self.project = None if d_project==d_model else XavierLinear(d_model, d_project)
        self.output = nn.Linear(d_project, n_classes, bias=False)

    def forward(self, seq, mask=None):
        out, mask = self.encoder(seq, mask)[0:2]
        out = self.project(out) if self.project is not None else out
        out = self.output(out)
        return out, mask

    def decode(self, seq, mask=None):
        logit, mask = self.forward(seq, mask)
        return torch.log_softmax(logit, -1), mask
