#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2021 Sebastian Wichmann
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import argparse

import torch

from pynn.util import load_object
from pynn.decoder.s2s import beam_search, beam_search_cache
from pynn.util.text import load_dict, write_hypo
from pynn.io.audio_seq import SpectroDataset
 
parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--model-dic', help='model dictionary', required=True)

parser.add_argument('--data-scp', help='path to data scp', required=True)
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')

parser.add_argument('--fp16', help='float 16 bits', action='store_true')
parser.add_argument('--batch-size', help='batch size', type=int, default=32)
parser.add_argument('--beam-size', help='beam size', type=int, default=10)
parser.add_argument('--output', help='output file', type=str, default='hypos/H_1_LV')
parser.add_argument('--format', help='output format', type=str, default='txt')
parser.add_argument('--space', help='space token', type=str, default='▁')

if __name__ == '__main__':
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    mdic = torch.load(args.model_dic)
    model = load_object(mdic['class'], mdic['module'], mdic['params'])
    model = model.to(device)
    model.load_state_dict(mdic['state'])
    model.eval()
    if args.fp16: model.half()
    
    reader = SpectroDataset(args.data_scp, mean_sub=args.mean_sub, fp16=args.fp16,
                            downsample=args.downsample)
    since = time.time()
    fout = open(f"{args.output}.{args.format}", 'w')
    with torch.no_grad():
        while True:
            seq, mask, utts = reader.read_batch_utt(args.batch_size)
            if not utts: break
            seq, mask = seq.to(device), mask.to(device)

            processedData = model.forward(seq, mask, softmax=True)[0].squeeze(1)
            hypos, scores = processedData.topk(args.beam_size, -1)
            hypos, scores = hypos.tolist(), scores.tolist()
            write_hypo(hypos, scores, fout, utts, None, space=args.space, output=args.format)
    fout.close()
    time_elapsed = time.time() - since
    print("  Elapsed Time: %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))
