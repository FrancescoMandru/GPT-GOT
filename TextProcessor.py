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

class TextProcessor(Dataset):

    def __init__(self, filepath, transform=None):


        ### Load data
        full_text = ''
        for file in os.listdir(filepath):
          if file.endswith(".txt"):
            print('\n Processing: ', file)
            text = open(filepath+file, 'r',  encoding="utf8").read()

            # Removing part of the characters
            text = re.sub("[^A-Za-z.,?!\" \n]+", '', text)

            # Lower case
            text = text.lower()

            # Isolating points and commas
            text = re.sub("[.]+", ' . ', text)
            text = re.sub("[,]+", ' , ', text)
            text = re.sub("[\"]+", ' ', text)
            text = re.sub("[;]+", ' ; ', text)
            text = re.sub("[?]+", ' ? ', text)
            text = re.sub("[!]+", ' ! ', text)

            full_text = full_text + text

        #print('\n\n Files processed, number of words: ', len(full_text))
        #avoid_words = ['q','w','e','r','t','y','u','o','p','s',
                       #'d','f','g','h','j','k','l','ill','il','z',
                       #'x','c','v','b','n','m']

        # Splitting text into a list of lists representing sentences of words
        text_sentences = [s.split() for s in full_text.splitlines() if len(s)>0]

        #text_sentences = [[s for s in sentence if s not in avoid_words]
                          #for sentence in text_sentences ]

        #text_sentences.append(['<pad>'])
        #print(text_sentences[-10:])

        print('\n\n Number of lines processed: ', len(text_sentences))

        self.full_text = full_text
        self.text_sentences = text_sentences

    # It creates the Word2Vec model
    def create_model(self):
        # build vocabulary and train model
        w2v_model = gensim.models.Word2Vec(self.text_sentences, workers=4, size=128,
                                           negative=7, hs=0,
                                           window=8, min_count=1,iter=20)
        return w2v_model
