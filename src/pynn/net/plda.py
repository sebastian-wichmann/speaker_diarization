# Copyright 2021 Sebastian Wichmann
# Licensed under the Apache License, Version 2.0 (the "License")


import torch
import torch.nn as nn
import torch.linalg as lin
import torch.nn.functional as F


class PLDA(nn.Module):
    def __init__(self, n_classes=1000, d_input=40, d_output=320, in_dropout=0.2):
        super(PLDA, self).__init__()


    def compute_scatter_matrices(self, src_seq, src_mask):
        count_per_class = torch.count_nonzero(src_mask)
        count = sum(count_per_class)

        mean_per_class = torch.sum(src_seq, 1) / count_per_class
        mean = torch.mean(mean_per_class)

        diff_src_mk = src_seq - mean_per_class.unsqueeze(1).expand(-1, src_seq.size(1), -1)
        diff_m_mk = mean_per_class - mean.unsqueeze(1).expand(-1, mean_per_class.size(1))

        S_w_k = torch.matmul(diff_src_mk.transpose(1, 2), diff_src_mk)
        S_b_k = torch.matmul(torch.matmul(diff_m_mk.unsqueeze(2), diff_m_mk. unsqueeze(1)), count_per_class)

        return (torch.sum(S_w_k, dim=0) / count), (torch.sum(S_b_k, dim=0) / count), mean, count, count_per_class


    def fit_model(self, src_seq, src_mask):
        # matching https://ravisoji.com/assets/papers/ioffe2006probabilistic.pdf

        S_w, S_b, mean, count, count_per_class = self.compute_scatter_matrices(src_seq, src_mask)
        assert(torch.max(count_per_class) == torch.min(count_per_class), "All Classes need same count of items")
        count_per_class = torch.min(count_per_class)

        # Calculate LDA Projection
        # S_b * w = lambda * S_w * w
        _, W = lin.eig(lin.solve(S_w, S_b))
        # transpose W
        Wt = torch.transpose(W, 0, 1)

        #
        lambda_b = torch.diagonal(torch.matmul(torch.matmul(Wt, S_b), W))
        lambda_w = torch.diagonal(torch.matmul(torch.matmul(Wt, S_w), W))

        # calculate A
        A = lin.solve(Wt, torch.sqrt((count_per_class / (count_per_class - 1)) * lambda_w).diag())

        # calculate psi
        psi = torch.clamp(((count_per_class - 1) / count_per_class) * (lambda_b / lambda_w).diag() - (1 / count_per_class), min=0)


        A_inverse = lin.inv(A)

        self.A_inverse = A_inverse
        self.mean = mean
        self.psi = psi
        return A_inverse, mean, psi

    def forward(self, src_seq, src_mask, tgt_seq, fit()):




