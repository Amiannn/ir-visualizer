import random 
import numpy  as np
import pandas as pd 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def preprocess_dataset(
        doc_datas, 
        query_datas,
        filter_out_class = "9",
        repeat_doc_datas = False,
        repeat_doc_datas_augument_scale = 10
    ):
    doc_dict = {d[-1]: d for d in doc_datas}
    
    datas = []
    d = doc_dict[filter_out_class]
    if repeat_doc_datas:
        for _ in range(1000):
            t, x1, x2, c = d
            x1 = x1 + ((np.random.rand(1)[0] - 0.5) / repeat_doc_datas_augument_scale)
            x2 = x2 + ((np.random.rand(1)[0] - 0.5) / repeat_doc_datas_augument_scale)
            data = [[t, x1, x2, c], d]
            datas.append(data)
    else:
        t, x1, x2, c = d
        data = [[t, x1, x2, c], d]
        datas.append(data)

    for q in query_datas:
        if q[-1] == filter_out_class:
            continue
        positive_ctx = doc_dict[q[-1]]
        data = [q, positive_ctx]
        datas.append(data)
    random.shuffle(datas)
    return datas

def preprocess_dataset_document_first(
        doc_datas, 
        query_datas,
        filter_out_class = "9",
        repeat_doc_datas = False,
        repeat_doc_datas_augument_scale = 10
    ):
    query_dict = {}
    for q in query_datas:
        c = q[-1]
        try:
            query_dict[c].append(q)
        except:
            query_dict[c] = [q]
    datas = []
    for k in range(100):
        for d in doc_datas:
            if d[-1] == filter_out_class or d[-1] not in query_dict:
                data = [[d], d]
            else:
                positive_queries = query_dict[d[-1]]
                data = [positive_queries, d]
            datas.append(data)

    # d = doc_dict[filter_out_class]
    # if repeat_doc_datas:
    #     for _ in range(1000):
    #         t, x1, x2, c = d
    #         x1 = x1 + ((np.random.rand(1)[0] - 0.5) / repeat_doc_datas_augument_scale)
    #         x2 = x2 + ((np.random.rand(1)[0] - 0.5) / repeat_doc_datas_augument_scale)
    #         data = [[t, x1, x2, c], d]
    #         datas.append(data)
    
    random.shuffle(datas)
    return datas

def normalize(datas):
    x = []
    for t, x1, x2, c in datas:
        x.append([float(x1), float(x2)])
    x = np.array(x)
    x = x - np.min(x)
    x = x / np.max(x)
    x = x - 0.5

    for i, data in enumerate(datas):
        data[1] = x[i][0]
        data[2] = x[i][1]
    return datas

def process_fn(queries, docs, device):
    _, q_x1, q_x2, c = queries
    _, d_x1, d_x2, c = docs

    q_x1 = [float(q) for q in q_x1]
    q_x2 = [float(q) for q in q_x2]

    d_x1 = [float(d) for d in d_x1]
    d_x2 = [float(d) for d in d_x2]
    
    c  = [int(d) for d in c]

    q_x = torch.tensor([q_x1, q_x2]).T
    d_x = torch.tensor([d_x1, d_x2]).T

    c = torch.tensor(c, dtype=torch.long)

    q_x = q_x.to(device)
    d_x = d_x.to(device)
    c   = c.to(device)
    return q_x, d_x, c

def collate_fn(batch_datas):
    q_prefix, q_x1, q_x2, q_c = [], [], [], []
    d_prefix, d_x1, d_x2, d_c = [], [], [], []

    for data in batch_datas:
        queries, docs = data
        queries = random.choice(queries)
        q_prefix.append(queries[0])
        q_x1.append(queries[1])
        q_x2.append(queries[2])
        q_c.append(queries[3])

        d_prefix.append(docs[0])
        d_x1.append(docs[1])
        d_x2.append(docs[2])
        d_c.append(docs[3])

    queries = [q_prefix, q_x1, q_x2, q_c]
    docs    = [d_prefix, d_x1, d_x2, d_c]
    return queries, docs