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

def train_epoch(net, device, dataloader, loss_fn, optimizer):

    # Training
    net.train()
    loss_list = []
    acc_list = []
    for (feature_t, label_t) in dataloader:

        # Move to GPU
        feature_t = feature_t.to(device)
        num_labels = label_t.shape[0] * label_t.shape[1]
        label_t = label_t.view(label_t.shape[0] * label_t.shape[1],)
        label_t = label_t.to(device)

        # Forward pass
        output, _ = net(feature_t)
        output = output.view(output.shape[1] * output.shape[0], output.shape[2])
        predict = torch.argmax(nn.functional.softmax(output),dim=1)
        correct = (predict == label_t).sum()
        loss = loss_fn(output, label_t)
        loss_list.append(loss.item())
        acc_list.append(correct.item() / num_labels)
        # Backward pass
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.25)
        loss.backward()
        optimizer.step()

    return np.mean(loss_list), np.mean(acc_list)



# Testing function
def test_epoch(net, device,  dataloader, loss_fn):
    # Validation
    net.eval()  # Evaluation mode (e.g. disable dropout)

    loss_list = []
    acc_list = []
    with torch.no_grad():  # No need to track the gradients
        for (feature_t, label_t) in dataloader:
            # Move to GPU
            feature_t = feature_t.to(device)
            num_labels = label_t.shape[0] * label_t.shape[1]
            label_t = label_t.view(label_t.shape[0] * label_t.shape[1],)
            label_t = label_t.to(device)

            # Forward pass
            output, _ = net(feature_t)
            output = output.view(output.shape[1] * output.shape[0], output.shape[2])
            predict = torch.argmax(nn.functional.softmax(output),dim=1)
            correct = (predict == label_t).sum()
            loss = loss_fn(output, label_t).item()
            acc_list.append(correct.item() / num_labels)
            loss_list.append(loss)

    return np.mean(loss_list), np.mean(acc_list)


# Word2Vec mapping from word to index
def word2idx(word, w2v_model):
    return w2v_model.wv.vocab[word].index

# Word2Vec mapping from word to index
def idx2word(idx, w2v_model):
    return w2v_model.wv.index2word[idx]

def check_vocab(text_words, w2v_model):

  new_text_words = []
  for word in text_words:
    if word not in w2v_model.wv.vocab:
      continue
    new_text_words.append(word)
  return new_text_words

def seed_processor(seed):
    ### Load data
    text = re.sub("[^A-Za-z.,?!\" \n]+", '', seed)
    # Lower case
    text = text.lower()
    # Isolating points and commas
    text = re.sub("[.]+", ' . ', text)
    text = re.sub("[,]+", ' , ', text)
    text = re.sub("[\"]+", ' ', text)
    text = re.sub("[;]+", ' ; ', text)
    text = re.sub("[?]+", ' ? ', text)
    text = re.sub("[!]+", ' ! ', text)
    # Splitting text into a list of lists representing sentences of words
    text_sentences = [s.split() for s in text.splitlines() if len(s)>0]
    return text_sentences[0]
