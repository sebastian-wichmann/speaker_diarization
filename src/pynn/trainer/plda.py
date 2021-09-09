# Copyright 2021 Sebastian Wichmann
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import math
from os import path

import torch
from torch.cuda.amp import autocast

from . import EpochPool, ScheduledOptim
from . import cal_ce_loss, load_last_chkpt, save_last_chkpt

def train_model(model, dataset, device, cfg, fp16=False, dist=False):
    ''' Start training '''
    model_path = cfg['model_path']

    #classes, data = datasets
    dataset.initialize()
    
    start = time.time()
    model.train()
    
    #data_len = len(dataset)
    data = dataset.get()

    # data = map(lambda x: x.to(device), data)

    try:
        # forward
        with autocast(enabled=fp16):
            model(data)
    except RuntimeError as err:
        if 'CUDA out of memory' in str(err):
            print('    WARNING: ran out of memory on GPU')
            torch.cuda.empty_cache()
        raise err

    print(f"  duration {(time.time()-start) / 60:3.3f}")

    model_file = path.join(model_path, 'plda.pt')
    torch.save(model.state_dict(), model_file)