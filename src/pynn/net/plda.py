# Copyright 2021 Sebastian Wichmann
# Licensed under the Apache License, Version 2.0 (the "License")


import torch
import torch.nn as nn
import torch.linalg as lin
import torch.nn.functional as F
import math


class PLDA(nn.Module):
    def __init__(self, dropout=0.2, dropconnect=0.2):
        super(PLDA, self).__init__()

        self.dropout_net = nn.Dropout(dropout)

    def compute_scatter_matrices(self, src_seq):
        count_classes = src_seq.size()[0]
        count_per_class = src_seq.size()[1]
        count_feature = src_seq.size()[2]
        count = src_seq.size()[0] * count_per_class

        mean_per_class = torch.mean(src_seq, 1)
        mean = torch.mean(src_seq, (0, 1))

        diff_src_mk = src_seq - mean_per_class.unsqueeze(1).expand(-1, src_seq.size()[1], -1)
        diff_m_mk = mean_per_class - mean.unsqueeze(0).expand(mean_per_class.size()[0], -1)


        S_w_k = torch.bmm(diff_src_mk.view(count_classes * count_per_class, count_feature, 1), diff_src_mk.view(count_classes * count_per_class, 1, count_feature))
        S_b_k = count_per_class * torch.bmm(diff_m_mk.view(count_classes, count_feature, 1), diff_m_mk.view(count_classes, 1, count_feature))
        """
        S_w_k = None
        S_b_k = None
        for classes in range(src_seq.size()[0]):
            S_w_k_tmp = torch.matmul(diff_src_mk[classes].transpose(0, 1), diff_src_mk[classes]).unsqueeze(0)
            S_b_k_tmp = torch.matmul(diff_m_mk[classes].unsqueeze(1), diff_m_mk[classes].unsqueeze(0)).unsqueeze(0)

            if S_w_k is None:
                S_w_k = S_w_k_tmp
            else:
                S_w_k = torch.cat((S_w_k, S_w_k_tmp), 0)

            if S_b_k is None:
                S_b_k = S_b_k_tmp
            else:
                S_b_k = torch.cat((S_w_k, S_b_k_tmp), 0)
        """
        return (torch.sum(S_w_k, dim=0) / count), (torch.sum(S_b_k, dim=0) / count), mean, count, count_per_class

    def fit_model(self, src_seq):
        # matching https://ravisoji.com/assets/papers/ioffe2006probabilistic.pdf

        S_w, S_b, mean, count, count_per_class = self.compute_scatter_matrices(src_seq)

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

        # calculate psi | diagonal Matrix
        psi = torch.clamp(
            ((count_per_class - 1) / count_per_class) * (lambda_b / lambda_w).diag() - (1 / count_per_class), min=0)
        psi_diag = psi.diag()

        A_inverse = lin.inv(A)

        self.A_inverse = A_inverse
        self.mean = mean
        self.psi_diag_full = psi_diag
        self.psi_diag = psi_diag
        return A_inverse, mean, psi_diag

    def reduce_dimension(self, dimension: int) -> bool:
        if dimension > self.psi_diag_full.size()[0]:
            self.psi_diag = self.psi_diag_full
            return False
        else:
            self.psi_diag = torch.zeros(self.psi_diag.size())
            vectors, index = self.psi_diag_full.topk(dimension, -1)
            self.psi_diag[index] = vectors
            return True

    def calculate_probability_single(self, vector, mean, deviation_inv): #TODO: deviation_inv wrong maybe not inv?
        vm = vector - mean
        exp = torch.matmul(torch.matmul(-vm, deviation_inv.diag), vm)
        divisor = torch.sqrt(((2 * math.pi) ** vector.size()[0]) * lin.norm(deviation_inv))
        return torch.exp(exp) / divisor

    def calculate_probability_multi(self, v1, v2, deviation_inv):  #TODO: deviation_inv wrong maybe not inv?
        v = torch.cat(v1, v2)
        mean = torch.mean(v, 0)

        v1_size, v2_size =  v1.size()[0], v2.size()[0]
        size = v1_size + v2_size

        to_mult = []
        for t in range(deviation_inv.size()[0]):
            tmp = deviation_inv[t] + (1 / size)

            exp = (-1) * (((mean[t] ** 2) / (2 * tmp)) + sum([ ((v[i][t] - mean[t]) ** 2) for i in range(size) ]) / 2) #TODO: replace vector multiplikation
            divisor = math.sqrt(((math.pi * 2) ** size) * tmp)
            to_mult.append((1 / divisor) * torch.exp(exp))
        
        return math.prod(to_mult)

    def forward(self, src_seq, src_mask, tgt_seq):
        seq = self.dropout_net(src_seq)

        if self.training:
            self.fit_model(seq, src_mask)

        latent = torch.matmul(self.A_inverse, (seq - self.mean))

        cluster = [ [i] for i in range(0, latent.size()[0]) ]
        l = 0
        l_new = 0

        while l <= l_new:
            l = l_new

            max_cluster = None
            cluster_ids = (None, None)

                
            for x in len(cluster):
                cluster_1 = cluster[x]
                for y in range(x + 1, len(cluster)):
                    cluster_2 = cluster[y]
                    if  (len(cluster_1) <= 1) and (len(cluster_2) <= 1):
                        probability = self.calculate_probability_single(latent[cluster_1], latent[cluster_2], self.psi_diag)
                    else:
                        probability = self.calculate_probability_multi(latent[cluster_1], latent[cluster_2], self.psi_diag)
                    
                    if (max_cluster is None) or (max_cluster < probability):
                        max_cluster = probability
                        cluster_ids = (x, y)
            
            # join cluster
            dest, source = cluster_ids
            cluster[dest] = cluster[dest] + cluster[source]
            del cluster[source]

            l_new += max_cluster
            
        return cluster
