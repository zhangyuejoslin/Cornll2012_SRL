import math
import numpy as np
import operator
import os
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
from .Highway import Highway
sys.path.append('../')
from config.global_config import CONFIG


import utils
cfg = CONFIG()
print(cfg)

class SRL_Model(torch.nn.Module):
    def __init__(self, pretrain_emb, tagset_size, is_test):
        super(SRL_Model, self).__init__()

        ### word embedding
        vocab_len = len(pretrain_emb)
        word_embeddings_dim = len(pretrain_emb[0])
        self.embeddings = nn.Embedding(vocab_len, word_embeddings_dim, padding_idx=vocab_len-1)
        if is_test is False:
            self.embeddings.weight.data.copy_(torch.from_numpy(np.array(pretrain_emb)))
            self.embeddings.weight.requires_grad = True
        else:
            self.embeddings.weight.requires_grad = False

        ## lstm
        self.lstm = nn.LSTM(word_embeddings_dim, cfg.lstm_hidden_dim, num_layers=cfg.num_lstm_layers, bidirectional=True)

        ## lstm highway gate
        self.highway_gates = Highway(cfg.lstm_hidden_dim, 1, f=torch.nn.functional.relu)

        # final layer
        self.hidden2tag = nn.Linear(cfg.lstm_hidden_dim, tagset_size)
    
    def forward(self, sentence, sen_mask):
        x = self.embeddings(sentence).unsqueeze(1)
        hidden_state, _ = self.lstm(x)
        hidden_state_highway_out = self.highway_gates(hidden_state)
        logits = self.hidden2tag(hidden_state_highway_out)
        return 0
    