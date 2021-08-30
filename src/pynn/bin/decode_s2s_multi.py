#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import argparse

import torch

from pynn.util import load_object
from pynn.decoder.s2s import beam_search, beam_search_cache
from pynn.util.text import load_dict, write_hypo
from pynn.io.audio_seq import SpectroDataset
from pynn.net.ensemble import Ensemble
 
parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--model-dic', type=str, action='append', nargs='+')
parser.add_argument('--lm-dic', help='language model dictionary', default=None)
parser.add_argument('--lm-scale', help='language model scale', type=float, default=0.5)

parser.add_argument('--dict', help='dictionary file', default=None)
parser.add_argument('--word-dict', help='word dictionary file', default=None)
parser.add_argument('--data-scp', help='path to data scp', required=True)
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')

parser.add_argument('--state-cache', help='caching encoder states', action='store_true')
parser.add_argument('--batch-size', help='batch size', type=int, default=32)
parser.add_argument('--beam-size', help='beam size', type=int, default=10)
parser.add_argument('--max-len', help='max len', type=int, default=100)
parser.add_argument('--fp16', help='float 16 bits', action='store_true')
parser.add_argument('--len-norm', help='length normalization', action='store_true')
parser.add_argument('--output', help='output file', type=str, default='hypos/H_1_LV.ctm')
parser.add_argument('--format', help='output format', type=str, default='ctm')
parser.add_argument('--space', help='space token', type=str, default='▁')

if __name__ == '__main__':
    args = parser.parse_args()

    dic, word_dic = load_dict(args.dict, args.word_dict)

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    models = []
    for dic_path in args.model_dic:
        sdic = torch.load(dic_path[0])
        model = load_object(sdic['class'], sdic['module'], sdic['params'])
        model = model.to(device)
        model.load_state_dict(sdic['state'])
        model.eval()
        if args.fp16: model.half()
        models.append(model)
    model = Ensemble(models)
    if args.fp16: model.half()

    lm = None
    if args.lm_dic is not None:
        mdic = torch.load(args.lm_dic)
        lm = load_object(mdic['class'], mdic['module'], mdic['params'])
        lm = lm.to(device)
        lm.load_state_dict(mdic['state'])
        lm.eval()
        if args.fp16: lm.half()
        
    reader = SpectroDataset(args.data_scp, mean_sub=args.mean_sub, fp16=args.fp16,
                            downsample=args.downsample)
    since = time.time()
    fout = open(args.output, 'w')
    decode_func = beam_search_cache if args.state_cache else beam_search
    with torch.no_grad():
        while True:
            seq, mask, utts = reader.read_batch_utt(args.batch_size)
            if not utts: break
            seq, mask = seq.to(device), mask.to(device)
            hypos, scores = decode_func(model, seq, mask, device, args.beam_size,
                                        args.max_len, len_norm=args.len_norm, lm=lm, lm_scale=args.lm_scale)
            hypos, scores = hypos.tolist(), scores.tolist()
            write_hypo(hypos, scores, fout, utts, dic, word_dic, args.space, args.format)
    fout.close()
    time_elapsed = time.time() - since
    print("  Elapsed Time: %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))
