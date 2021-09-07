# Copyright 2021 Sebastian Wichmann
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import math
from os import path

import torch
from torch.cuda.amp import autocast

from . import EpochPool, ScheduledOptim
from . import cal_ce_loss, load_last_chkpt, save_last_chkpt

def train_model(model, datasets, epochs, device, cfg, fp16=False, dist=False):
    ''' Start training '''
    model_path = cfg['model_path']

    data = datasets
    data.initialize(b_input, b_sample)
    
    start = time.time()
    model.train()
    
    data_len = len(data)
    loader = data.create_loader()

    for batch_i, batch in enumerate(loader):
        # prepare data
        src_seq, src_mask, tgt_seq = map(lambda x: x.to(device), batch)
        last = (batch_i == data_len)
        n_seq += tgt_seq.size(0)

        try:
            # forward
            with autocast(enabled=fp16):
                model(src_seq, src_mask, tgt_seq, encoding=not sampling)[0]
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                print('    WARNING: ran out of memory on GPU at %d' % n_seq)
                torch.cuda.empty_cache(); continue
            raise err

    print(f"  duration {(time.time()-start)/60):3.3f}")

    model_file = path.join(model_path, 'plda.pt')
    torch.save(model.state_dict(), model_file)