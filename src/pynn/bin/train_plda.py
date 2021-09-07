#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2021 Sebastian Wichmann
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pynn.util import save_object_param
from pynn.net.plda import PLDA
from pynn.bin import print_model, train_plda_model

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--train-embeddings', help='path to train embeddings', required=True)
parser.add_argument('--train-classes', help='path to train classes', required=True)

parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--dropconnect', type=float, default=0.)

parser.add_argument('--model-path', help='model saving path', default='model')

parser.add_argument('--fp16', help='fp16 or not', action='store_true')


def create_model(args, device):
    params = {
        'dropout': args.dropout,
        'dropconnect': args.dropconnect}
    model = PLDA(**params)
    save_object_param(model, params, args.model_path + '/model.cfg')
    return model


def train(device, args):
    model = create_model(args, device)
    print_model(model)
    train_plda_model(model, args, device)


def train_distributed(device, gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=device, world_size=gpus)
    torch.manual_seed(0)

    model = create_model(args, device)
    if device == 0:
        print_model(model)
    train_plda_model(model, args, device, gpus)

    dist.destroy_process_group()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    if torch.cuda.device_count() > 1:
        gpus = torch.cuda.device_count()
        print('Training with distributed data parallel. Number of devices: %d' % gpus)
        mp.spawn(train_distributed, nprocs=gpus, args=(gpus, args), join=True)
    else:
        device = 0 if torch.cuda.is_available() else torch.device('cpu')
        train(device, args)
