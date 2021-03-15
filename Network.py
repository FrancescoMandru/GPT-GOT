import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torch.utils.data as data
import gensim
from gensim.models import Word2Vec
import re
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

class Network(nn.Module):

    def __init__(self, pretrained_weights, hidden_units, layers_num, dropout_prob=0):
        # Call the parent init function (required!)
        super().__init__()

        # Define Embedding layer
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_weights))

        vocab_size, emdedding_size = pretrained_weights.shape

        # Define recurrent layer
        self.rnn = nn.LSTM(input_size=emdedding_size,
                           hidden_size=hidden_units,
                           num_layers=layers_num,
                           dropout=dropout_prob,
                           batch_first=True)
        # Define output layer
        self.out = nn.Linear(hidden_units, vocab_size)

    def forward(self, x, state=None):

        # EMBEDDING
        x = self.embedding(x)

        # LSTM
        x, rnn_state = self.rnn(x, state)

        # Linear layer
        x = self.out(x)

        return x, rnn_state
