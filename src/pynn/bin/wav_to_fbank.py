#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse
import multiprocessing
import threading
import numpy as np
from scipy.io import wavfile

from pynn.util import audio
from pynn.io import kaldi_io

queue_end_string = "::end::"

def write_ark_thread(segs, out_ark, out_scp, queue_default, queue_error, args):
    fbank_mat = audio.filter_bank(args.sample_rate, args.nfft, args.fbank)
    cache_wav = ''
    
    ark_file = open(out_ark, 'wb')
    scp_file = open(out_scp, 'w')
    for seg in segs:
        tokens = seg.split()
        
        if len(tokens) == 1: tokens.insert(0, '')
        if len(tokens) == 2: tokens.extend(['0.0', '0.0'])
        seg_name, wav, start, end = tokens[:4]

        start, end = float(start), float(end)
        
        if args.wav_path is not None:
            wav = wav if wav.endswith('.wav') else wav + '.wav'
            wav = os.path.join(args.wav_path, wav)

        if seg_name == '':
            seg_name = os.path.basename(wav)[:-4]

        if cache_wav != wav:
            if not os.path.isfile(wav):
                queue_default.put(f'File {wav} does not exist')
                continue
            try:
                sample_rate, signal = wavfile.read(wav)
            except ValueError as ex:
                queue_error.put(f'Value error occurred while reading file {wav}\n'
                                f'Error parameter: {ex.args}\n'
                                f'Default message: {ex}')


            if sample_rate != args.sample_rate:
                queue_default.put(f'Wav {wav} is not in desired sample rate')
                continue
            cache_wav = wav

        end = float(len(signal)) / sample_rate if end <= 0. else end
        if args.seg_info:
            seg_name = '%s-%06.f-%06.f' % (seg_name, start*100, end*100)
        start, end = int(start * sample_rate), int(end * sample_rate)
        if start >= len(signal) or start >= end >= 0:
            queue_default.put(f'Wrong segment {seg_name}')
            continue

        feats = audio.extract_fbank(signal[start:end], fbank_mat, sample_rate=sample_rate, nfft=args.nfft)
        if len(feats) > args.max_len or len(feats) < args.min_len:
            continue
        if args.mean_norm:
            feats = feats - feats.mean(axis=0, keepdims=True)
        if args.fp16:
            feats = feats.astype(np.float16) 

        dic = {seg_name: feats}
        #kaldi_io.write_ark(out_ark, dic, out_scp, append=True)
        kaldi_io.write_ark_file(ark_file, scp_file, dic)
    ark_file.close()
    scp_file.close()

def write_queue_to_file(queue, file = None):
    if file is not None:
        with open(file, 'w') as out_handle:
            while True:
                value = queue.get()
                if value == queue_end_string:
                    break
                out_handle.write(f'{value}\n')
                print(value)
    else:
        while True:
            value = queue.get()
            if value == queue_end_string:
                break
            print(value)

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--seg-desc', help='input segment description file', required=True)
parser.add_argument('--seg-info', help='append timestamp suffix to segment name', action='store_true')
parser.add_argument('--wav-path', help='path to wav files', type=str, default=None)
parser.add_argument('--sample-rate', help='sample rate', type=int, default=16000)
parser.add_argument('--fbank', help='number of filter banks', type=int, default=40)
parser.add_argument('--nfft', help='number of FFT points', type=int, default=256)
parser.add_argument('--max-len', help='maximum frames for a segment', type=int, default=10000)
parser.add_argument('--min-len', help='minimum frames for a segment', type=int, default=4)
parser.add_argument('--mean-norm', help='mean substraction', action='store_true')
parser.add_argument('--fp16', help='use float16 instead of float32', action='store_true')
parser.add_argument('--output', help='output file', type=str, default='data')
parser.add_argument('--jobs', help='number of parallel jobs', type=int, default=1)
parser.add_argument('--log-error', help='output file for error messages', type=str, default=None)
parser.add_argument('--log-default', help='output file for error messages', type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()

    manager = multiprocessing.Manager()
    out_queue = manager.Queue()
    error_queue = manager.Queue()

    default_thread = threading.Thread(target=write_queue_to_file, args=(out_queue, args.log_default))
    default_thread.start()
    error_thread = threading.Thread(target=write_queue_to_file, args=(error_queue, args.log_error))
    error_thread.start()

    segs = [line.rstrip('\n') for line in open(args.seg_desc, 'r')]
    size = len(segs) // args.jobs
    jobs = []
    j = 0
    for i in range(args.jobs):
        l = len(segs) if i == (args.jobs-1) else j+size
        sub_segs = segs[j:l]
        j += size
        out_ark = '%s.%d.ark' % (args.output, i)
        out_scp = '%s.%d.scp' % (args.output, i)
         
        process = multiprocessing.Process(
                target=write_ark_thread, args=(sub_segs, out_ark, out_scp, out_queue, error_queue, args))
        process.start()
        jobs.append(process)
    
    for job in jobs: job.join()

    out_queue.put(queue_end_string)
    error_queue.put(queue_end_string)
    default_thread.join()
    error_thread.join()
    
