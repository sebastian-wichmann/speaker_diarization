# Copyright 2021 Sebastian Wichmann
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import freeze_module
from pynn.net.tdnn import ConvTDNN, LinearTDNN
from pynn.net.statistics_pooling_layer import StatisticsPoolingLayer


class X_VectorFrame(nn.Module):
    def __init__(self, d_input, d_output, use_cnn=False, stride=1, dropout=0.0):
        super().__init__()

        tdnn_config_list = [(5, 1, d_input, 512),
                            (3, 2, 512, 512),
                            (3, 3, 512, 512),
                            (1, 1, 512, 512),
                            (1, 1, 512, d_output)]

        layer_list = nn.ModuleList()

        # For some reason this doens't work
        #  -> limited to only "Linear" option
        # if use_cnn:
        #    for conf in tdnn_config_list:
        #        layer_list.append(ConvTDNN(conf[2], conf[3], conf[0], stride, conf[1], False, dropout))
        # else:
        for conf in tdnn_config_list:
            layer_list.append(LinearTDNN(conf[2], conf[3], conf[0], stride, conf[1], False, dropout))

        self.tdnn = nn.Sequential(*layer_list)

    def forward(self, seq):

        return self.tdnn(seq)


class X_VectorSegment(nn.Module):
    def __init__(self, d_input, d_output, n_speaker, dropout=0.0):
        super().__init__()

        layer_list = nn.ModuleList()

        layer_list.append(StatisticsPoolingLayer(d_input, calc_dim=1))
        if dropout:
            layer_list.append(nn.Dropout(p=dropout))
        layer_list.append(nn.Linear(d_input * 2, d_output))
        if dropout:
            layer_list.append(nn.Dropout(p=dropout))
        layer_list.append(nn.Linear(d_output, n_speaker))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, enc_out):
        return self.layers.forward(enc_out)


class X_Vector(nn.Module):
    def __init__(self, n_speaker=1000, d_input=20, d_enc=1500, n_enc=6, d_dec=512, n_dec=2,
                 unidirect=False, incl_win=0, d_emb=0, d_project=0, n_head=8, shared_emb=False,
                 time_ds=1, use_cnn=False, freq_kn=3, freq_std=2, enc_dropout=0.2, enc_dropconnect=0.,
                 dec_dropout=0.1, dec_dropconnect=0., emb_drop=0., pack=True):
        self.n_speaker = n_speaker

        super(X_Vector, self).__init__()

        self.frame_module = X_VectorFrame(d_input, d_enc, use_cnn=use_cnn, dropout=enc_dropconnect)

        self.statPooling = StatisticsPoolingLayer(d_enc, calc_dim=1)

        self.a_layer = nn.Sequential(nn.Dropout(p=dec_dropout), nn.Linear(d_enc * 2, d_dec)) if dec_dropout \
            else nn.Linear(d_enc * 2, d_dec)

        self.b_layer = nn.Sequential(nn.Dropout(p=dec_dropout), nn.Linear(d_dec, 300)) if dec_dropout \
            else nn.Linear(d_dec, 300)

        self.preSoftmax = nn.Sequential(nn.Dropout(p=dec_dropout), nn.Linear(300, n_speaker)) if dec_dropout \
            else nn.Linear(300, n_speaker)

        self.softmax = nn.Softmax(-1)

    def freeze(self, mode=0):
        if mode == 1:
            freeze_module(self.encoder)
            print("freeze the encoder")

    def forward(self, src_seq, src_mask=[], tgt_seq=[], encoding=True, speaker_number=True, a_embedings=False, b_embedings=False, softmax=False):
        out = self.frame_module(src_seq)
        tmp = []

        out = self.statPooling(out)

        if a_embedings or b_embedings or speaker_number or softmax:
            out = self.a_layer(out)
        if a_embedings:
            tmp.append(out)

        if b_embedings or speaker_number or softmax:
            out = self.b_layer(out)
        if b_embedings:
            tmp.append(out)

        if speaker_number or softmax:
            out = self.preSoftmax(out)
        if softmax:
            out = self.softmax(out)

        if speaker_number or softmax:
            tmp.append(out)

        if len(tmp) > 1:
            max_value = max(list(map(lambda x: x.size()[-1], tmp)))
            tmp = list(map(lambda x:
                           torch.cat((x, torch.zeros(x.size()[0], max_value - x.size()[-1], device=x.device)), -1)
                           .unsqueeze(1), tmp))
            out = torch.cat(tmp, 1)
        else:
            out = out.unsqueeze(1)
        return out, src_seq, src_mask
