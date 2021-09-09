# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from . import smart_open

class TextSeqDataset(Dataset):
    def __init__(self, path, sek=True, threads=1, verbose=True):
        self.path = path
        self.sek = sek

        self.threads = threads
        self.seqs = []

        self.verbose = verbose
        self.rank = 0
        self.parts = 1
        self.epoch = -1

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        return self.seqs[index]

    def print(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)

    def partition(self, rank, parts):
        self.rank = rank
        self.parts = parts

    def set_epoch(self, epoch):
        self.epoch = epoch
       
    def initialize(self, b_input=0, b_sample=256):
        seqs = []
        for line in smart_open(self.path, 'rt'):
            tokens = line.split()
            if len(tokens) < 2: continue
            #seq_id = tokens[0]
            seq = [int(token) for token in tokens]
            seq = [1] + [el+2 for el in seq] + [2] if self.sek else seq
            seqs.append(seq)

        self.print('%d label sequences loaded.' % len(seqs))
        self.seqs = seqs

        self.print('Creating batches.. ', end='')
        self.batches = self.create_batch(b_input, b_sample)
        self.print('Done.')

    def create_loader(self):
        batches = self.batches.copy()
        if self.epoch > -1:
            random.seed(self.epoch)
            random.shuffle(batches)
        if self.parts > 1:
            l = (len(batches) // self.parts) * self.parts
            batches = [batches[j] for j in range(self.rank, l, self.parts)]

        loader = DataLoader(self, batch_sampler=batches, collate_fn=self.collate_fn,
                            num_workers=self.threads, pin_memory=False)
        return loader

    def create_batch(self, b_input, b_sample):
        if b_input <= 0: b_input = b_sample*1000

        lst = [(j, len(seq)) for j, seq in enumerate(self.seqs)]
        lst = sorted(lst, key=lambda e : e[1])

        s, j, step = 0, 4, 4
        batches = []
        while j <= len(lst):
            bs = j - s
            if lst[j-1][1]*bs < b_input and bs < b_sample:
                j += step
                continue
            if bs > 8: j = s + (bs // 8) * 8
            batches.append([idx for idx, _ in lst[s:j]])
            s = j
            j += step
        if s < len(lst): batches.append([idx for idx, _ in lst[s:]])
        return batches

    def collate_fn(self, batch):
        max_len = max(len(inst) for inst in batch)
        seqs = np.array([inst + [0] * (max_len - len(inst)) for inst in batch])
        return torch.LongTensor(seqs)

class TextPairDataset(Dataset):
    def __init__(self, path, src_sek=True, tgt_sek=True, sort_src=False, threads=1, verbose=True):
        self.path = path
        self.src_sek = src_sek
        self.tgt_sek = tgt_sek

        self.sort_src = sort_src
        self.threads = threads
        self.seqs = []

        self.verbose = verbose
        self.rank = 0
        self.parts = 1
        self.epoch = -1

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        return self.seqs[index]

    def print(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)

    def partition(self, rank, parts):
        self.rank = rank
        self.parts = parts

    def set_epoch(self, epoch):
        self.epoch = epoch
       
    def initialize(self, b_input=0, b_sample=256):
        seqs = []
        for line in smart_open(self.path, 'rt'):
            tokens = line.split()
            sp = tokens.index('|')
            if sp < 1: continue
            
            src = [int(token) for token in tokens[:sp]]
            src = [1] + [el+2 for el in src] + [2] if self.src_sek else src

            tgt = [int(token) for token in tokens[sp+1:]]
            tgt = [1] + [el+2 for el in tgt] + [2] if self.tgt_sek else tgt
            seqs.append([src, tgt])
        
        self.print('%d label sequences loaded.' % len(seqs))
        self.seqs = seqs

        self.print('Creating batches.. ', end='')
        self.batches = self.create_batch(b_input, b_sample)
        self.print('Done.')

    def create_loader(self):
        batches = self.batches.copy()
        if self.epoch > -1:
            random.seed(self.epoch)
            random.shuffle(batches)
        if self.parts > 1:
            l = (len(batches) // self.parts) * self.parts
            batches = [batches[j] for j in range(self.rank, l, self.parts)]

        loader = DataLoader(self, batch_sampler=batches, collate_fn=self.collate_fn,
                            num_workers=self.threads, pin_memory=False)
        return loader

    def create_batch(self, b_input, b_sample):
        if b_input <= 0: b_input = b_sample*1000

        lst = [(j, len(seq[1])) for j, seq in enumerate(self.seqs)]
        lst = sorted(lst, key=lambda e : e[1])

        s, j, step = 0, 4, 4
        batches = []
        while j <= len(lst):
            bs = j - s
            if lst[j-1][1]*bs < b_input and bs < b_sample:
                j += step
                continue
            if bs > 8: j = s + (bs // 8) * 8
            batches.append([idx for idx, _ in lst[s:j]])
            s = j
            j += step
        if s < len(lst): batches.append([idx for idx, _ in lst[s:]])
        return batches

    def collate_fn(self, batch):
        if self.sort_src:
            batch = sorted(batch, key=lambda e : -len(e[0]))
        src, tgt = zip(*batch)
        
        max_len = max(len(inst) for inst in src)
        src = np.array([inst + [0] * (max_len - len(inst)) for inst in src])
        src = torch.LongTensor(src)
        mask = src.gt(0)
        
        max_len = max(len(inst) for inst in tgt)
        tgt = np.array([inst + [0] * (max_len - len(inst)) for inst in tgt])
        tgt = torch.LongTensor(tgt)
        
        return src, mask, tgt


class TextFeatureDataset(TextSeqDataset):
    def __init__(self, path_data, path_classes = None, threads=1, seed=-1, verbose=True, threshold = 20, path_preproccesed = None):
        self.path_data = path_data
        self.path_classes = path_classes

        self.path_preproccesed = path_preproccesed

        self.threshold = threshold

        self.threads = threads
        self.seqs = []

        self.seed = seed

        self.verbose = verbose
        self.utt_feature = None
        

    def _load_features(self, features):
        return torch.HalfTensor([float(x) for x in features])

       
    def initialize(self):
        if self.utt_feature is not None:
            return

        # open features
        loaded_data = {}
        for line in smart_open(self.path_data, 'r'):
            if line.startswith('#'): continue
            tokens = line.strip().split(' ')
            if len(tokens) < 2: continue
            utt_id, *features = tokens

            if utt_id == '': continue
            loaded_data[utt_id] = (utt_id, self._load_features(features))
        
        if self.path_classes is not None:
            # load classes
            # filter to match all existing features 
            loaded_classes = {}
            for line in smart_open(self.path_classes, 'r'):
                if line.startswith('#'): continue
                tokens = line.strip().split(' ')
                if len(tokens) < 2: continue
                utt_id, class_data = tokens

                if utt_id == '' or utt_id not in loaded_data.keys(): continue
                loaded_classes[utt_id] = (utt_id, int(class_data))
            
            # filter to match all existing features
            to_delete = []
            for utt_id in loaded_data.keys():
                if utt_id not in loaded_classes.keys():
                    to_delete.append(utt_id)
            for item in to_delete:
                del loaded_data[item]
            
            assert len(loaded_data) == len(loaded_classes), "Loaded classes, and loaded features don't match"

            # match class information to feature
            data = []
            loaded_data = sorted(loaded_data.values(), key=lambda x: x[0])
            loaded_classes = sorted(loaded_classes.values(), key=lambda x: x[0])
            for (utt_1, feats), (_, class_data) in zip(loaded_data, loaded_classes):
                data.append((utt_1, class_data, feats))
            #loaded_data_tmp = [(utt_1, class_data, feats) for utt_1, feats in loaded_data.values() for utt_2, class_data in loaded_classes.values() if utt_1 == utt_2]
            #loaded_data = sorted(loaded_data, key=lambda x: (x[1], x[0]))
            loaded_data = data

            # group data by class
            grouped = {}
            for utt, class_data, feats in loaded_data:
                data = (utt, feats)
                if class_data not in grouped:
                    grouped[class_data] = [data]
                else:
                    grouped[class_data].append(data)
            
            # scramble classes lists:
            random.seed(self.seed)
            for key in grouped:
                random.shuffle(grouped[key])
            loaded_data = list(grouped.values())

            # remove classes with less then $threshold occurrence
            loaded_data = [x for x in loaded_data if len(x) >= self.threshold]

            # normalize occurrence of each class to min occurrences
            min_value = min(map(lambda x: len(x), loaded_data))
            for index in range(len(loaded_data)):
                del loaded_data[index][min_value:] 
        else:
            loaded_data = list(loaded_data.values())

        self.seqs = loaded_data
        self.print(f'{len(loaded_data)} feature sequences loaded.')
        self.print('Done.')

    def get(self):
        to_return = None

        for classes in self.seqs:
            class_vectors = None
            for _, tensor in classes:
                tensor = tensor.unsqueeze(0)
                if class_vectors is None:
                    class_vectors = tensor
                else:
                    class_vectors = torch.cat((class_vectors, tensor), 0)

            class_vectors = class_vectors.unsqueeze(0)
            if to_return is None:
                to_return = class_vectors
            else:
                to_return = torch.cat((to_return, class_vectors), 0)

        return to_return

