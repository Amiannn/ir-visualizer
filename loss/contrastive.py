import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn

def dot_product_scores(q_vectors, ctx_vectors):
    # row_vector: n1 x D, col_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r

def dist_scores(q_vectors, ctx_vectors):
    r = torch.cdist(q_vectors, ctx_vectors)
    return r * -1

def gaussian_ll(q_means, d_means, d_sigma, row="d"):
    # q_means: n1 x D, d_means: n2 x D, d_sigma: n2 x D, result n1 x n2
    n1 = q_means.shape[0]
    n2 = d_means.shape[0]
    q_means = q_means.unsqueeze(1)
    d_means = d_means.unsqueeze(1)
    d_sigma = d_sigma.unsqueeze(1)

    q_means = q_means.repeat(1, n1, 1)
    d_means = d_means.repeat(1, n1, 1).transpose(1, 0)
    d_sigma = d_sigma.repeat(1, n1, 1).transpose(1, 0)

    d_sigma_2 = torch.pow(d_sigma, 2)

    alpha = 100
    r = torch.sum((torch.log(2 * np.pi * d_sigma_2) + (torch.pow(q_means - d_means, 2) / (d_sigma_2))) * -0.5, dim=-1)
    # r = torch.sum(1 / (torch.pow(2 * np.pi * d_sigma_2, 0.5)) * torch.exp((torch.pow(q_means - d_means, 2) / d_sigma_2) * -0.5), dim=-1)
    return r

def contrastive_loss(c, z1, z2, z2_sigma=None, device='cpu', score_type="dist"):
    if score_type == "dot_product":
        score_fn = dot_product_scores
    elif score_type == "dist":
        score_fn = dist_scores
    elif score_type == "gaussian":
        score_fn = gaussian_ll

    n1 = c.shape[0]
    c  = c.unsqueeze(1)
    neg_mask = (c.repeat(1, n1) == c.repeat(1, n1).T).float()
    
    if score_type == "gaussian":
        scores = score_fn(z1, z2, z2_sigma)
        neg_mask_T = neg_mask.T

        log_softmax_scores   = F.log_softmax(scores,   dim=1)
        log_softmax_scores_T = F.log_softmax(scores.T, dim=1)
        loss     = torch.sum(-1 * (log_softmax_scores * neg_mask_T) / torch.sum(neg_mask_T, dim=1))
        loss_T   = torch.sum(-1 * (log_softmax_scores_T * neg_mask) / torch.sum(neg_mask,   dim=1))
        return (loss + loss_T) * 0.5
    else:
        scores = score_fn(z1, z2)
    

        log_softmax_scores = F.log_softmax(scores, dim=1)
        # neg_mask = torch.eye(log_softmax_scores.shape[0]).to(device)
        loss     = torch.sum(-1 * (log_softmax_scores * neg_mask) / torch.sum(neg_mask, dim=1))
        return loss