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

class SRL_Model(torch.nn.Module):
    def __init__(self, pretrain_emb, target_vocab, is_test):
        super(SRL_Model, self).__init__()

        ### word embedding
        vocab_len = len(pretrain_emb)
        word_embeddings_dim = len(pretrain_emb[0])
        # self.embeddings = nn.Embedding(vocab_len, word_embeddings_dim, padding_idx=0)
        self.embeddings = nn.Embedding(vocab_len, word_embeddings_dim)
        self.embeddings.weight.data.copy_(torch.from_numpy(np.array(pretrain_emb)))
        self.embeddings.weight.requires_grad = True

        ## lstm
        self.lstm = nn.LSTM(word_embeddings_dim, cfg.lstm_hidden_dim, num_layers=cfg.num_lstm_layers, bidirectional=True, batch_first=False)

        ## lstm highway gate
        self.highway_gates = Highway(cfg.lstm_hidden_dim*2, 1, f=torch.nn.functional.relu)

        # final layer
        self.hidden2tag = nn.Linear(cfg.lstm_hidden_dim*2, len(target_vocab))
    
    def forward(self, sentence, sen_mask):
        x = self.embeddings(sentence)
        sen_mask = sen_mask.transpose(0, 1)
        hidden_state, _ = self.lstm(x)
        hidden_state = sen_mask.unsqueeze(2) * hidden_state
        # hidden_state_highway_out = self.highway_gates(hidden_state)
        # hidden_state_highway_out = sen_mask.unsqueeze(2) * hidden_state_highway_out
        # logits = self.hidden2tag(hidden_state_highway_out)
        logits = self.hidden2tag(hidden_state)
        return logits

    def get_span_candidates(self, text_len, max_sentence_length, max_mention_width):
        """Get a list of candidate spans up to length W.
        Args:
            text_len: Tensor of [num_sentences,]
            max_sentence_length: Integer scalar.
            max_mention_width: Integer.
        """
        candidate_starts = 0
        candidate_ends = 0
        candidate_mask = 0
        return candidate_starts, candidate_ends, candidate_mask
    