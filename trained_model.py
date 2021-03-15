import numpy as np
import random
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
import argparse
warnings.filterwarnings("ignore")

from TextProcessor import TextProcessor
from Network import Network
from methods import train_epoch, test_epoch, word2idx, idx2word, check_vocab, seed_processor


word2vec = False
pretrain = True
training = False
testing = True
# Padd - 13817

# Checking for GPU devices
if torch.cuda.is_available():
  dev = "cuda:0"
  print("Device: cuda")
else:
  dev = "cpu"
  print("Device: cpu")

# Device used
device = torch.device(dev)
# Filepath
filepath = './books_dir/'

if word2vec:

    print('\n\n Word2Vec Startining...')
    # %% Initialize dataset
    dataset = TextProcessor(filepath)

    model = dataset.create_model()
    vocab_size, emdedding_size = model.wv.vectors.shape
    print('Vocabulary Size: ', vocab_size)
    print('Embedding Size: ', emdedding_size)
    print('Hidden layer weights matrix shape: ', model.wv.vectors.shape)
    model.save("word2vec.model")
    print("Model created and saved")

if training:

    # Reading the dataset
    dataset = TextProcessor(filepath)
    # Loading word2vec
    print('\n\n --- LOADING Word2Vec')
    w2v_model = Word2Vec.load('word2vec.model')
    # Processing dataset
    text_words = dataset.full_text.split()
    # Vocabulary pruning
    print('\n\n --- Vocabulary pruning')
    text_words = check_vocab(text_words,w2v_model)

    # Data pre-processing
    max_length_sentence = 30

    # Sentences of equal size
    dataset_X = []
    dataset_Y = []

    # Splitting text into sentences
    #for i in range(len(text_words) - max_length_sentence - 1):
      #s = text_words[i: i + max_length_sentence]
      #dataset_X.append([word2idx(word, w2v_model) for word in s[:-1]])  # features
      #dataset_Y.append([word2idx(word, w2v_model) for word in s[1:]])  # labels


    stride = 3
    for i in range(0, len(text_words) - max_length_sentence - 1, stride):
      s = text_words[i: i + max_length_sentence]
      dataset_X.append([word2idx(word, w2v_model) for word in s[:-1]])  # features
      dataset_Y.append([word2idx(word, w2v_model) for word in s[1:]])  # labels


    X_train, X_val, y_train, y_val = train_test_split(dataset_X,
                                                      dataset_Y,
                                                      test_size=0.30,
                                                      random_state=1204532)

    X_val, X_test, y_val, y_test = train_test_split(X_val,
                                                    y_val,
                                                    test_size=0.10,
                                                    random_state=1204532)

    print('\n\n ----- DATASET DIMENSION')
    print(' ----  Training size: ', len(X_train))
    print(' ----  Validation size: ', len(X_val))
    print(' ----  Testing size: ', len(X_test))


    torch_X_train = torch.LongTensor(X_train)
    torch_X_val = torch.LongTensor(X_val)
    torch_X_test = torch.LongTensor(X_test)
    torch_Y_train = torch.LongTensor(y_train)
    torch_Y_val = torch.LongTensor(y_val)
    torch_Y_test = torch.LongTensor(y_test)


    train_dataset = data.TensorDataset(torch_X_train, torch_Y_train)
    valid_dataset = data.TensorDataset(torch_X_val, torch_Y_val)
    test_dataset = data.TensorDataset(torch_X_test, torch_Y_test)

    batch_size = 512
    hidden_units = 512
    layers_num = 3
    dropout_prob = 0.10
    pretrained_weights = w2v_model.wv.vectors
    print(' ----  Batch Size: ', batch_size)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    net = Network(pretrained_weights, hidden_units, layers_num, dropout_prob)
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    print('\n\n ----- NETWORK SPECIFICATIONS')
    print(' ----  Hidden Units: ', hidden_units)
    print(' ----  Layers Number: ', layers_num)
    print(' ----  Dropout: ', dropout_prob)

    print('\n\n ----- NETWORK STARTING')

    net.to(device)

    # We will be using a simple Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # Epochs
    epochs = 10000
    epoch_checkpoint = 0
    # Load pre-trained weights
    loss_eph_train = []
    loss_eph_val = []
    acc_eph_train = []
    acc_eph_val = []
    if pretrain:

        checkpoint = torch.load('LSTMparams')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_checkpoint = checkpoint['epoch']
        loss_eph_train = checkpoint['loss_eph_train'][:-1]
        loss_eph_val = checkpoint['loss_eph_val'][:-1]
        acc_eph_train = checkpoint['acc_eph_train'][:-1]
        acc_eph_val = checkpoint['acc_eph_val'][:-1]

    for epoch in range( epoch_checkpoint, epochs ):
        loss_train, acc_train = train_epoch(net, device, dataloader=train_loader, loss_fn=criterion,
                                 optimizer=optimizer)
        loss_val, acc_val = test_epoch(net, device, dataloader=valid_loader, loss_fn=criterion)


        print('EPOCH N° {} - TRAIN LOSS {} - TRAIN ACC {} - VAL LOSS {} - VAL ACC {}'.format(epoch,
                                                                np.round(loss_train,4),
                                                                np.round(acc_train,4),
                                                                np.round(loss_val,4),
                                                                 np.round(acc_val,4)) )

        loss_eph_train.append(loss_train)
        acc_eph_train.append(acc_train)
        loss_eph_val.append(loss_val)
        acc_eph_val.append(acc_val)

        if epoch % 5 == 0 or epoch == epochs-1:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_eph_train': loss_eph_train,
                    'loss_eph_val': loss_eph_val,
                    'acc_eph_train': acc_eph_train,
                    'acc_eph_val': acc_eph_val}, 'LSTMparams')

if testing:

    batch_size = 512
    hidden_units = 512
    layers_num = 3
    dropout_prob = 0.10
    print('\n\n LOAD PRE-TRAINED WORD2VEC')
    w2v_model = Word2Vec.load('word2vec.model')
    pretrained_weights = w2v_model.wv.vectors

    net = Network(pretrained_weights, hidden_units, layers_num, dropout_prob)
    print('\n\n LOADING TRAINED NETWORK')
    checkpoint = torch.load('LSTMparams', map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    #loss_test, acc_test = test_epoch(net, device, dataloader=test_loader, loss_fn=criterion)
    #print('\n\n TEST LOSS: ', np.round(loss_test,4))
    #print('\n\n ACCURACY TEST: ', np.round(acc_test,4))

    print('\n\n GENERATING TEXT FROM SEED')
    parser = argparse.ArgumentParser(description='Person Identification Dataset Script')
    parser.add_argument('--seed', default='love is', type=str)
    parser.add_argument('--length', default=50, type=int)
    args = parser.parse_args()
    seed = args.seed
    length = args.length
    text_words = seed_processor(seed)
    ### Check the word
    text_words = check_vocab(text_words,w2v_model)
    if len(text_words)==0:
        print("No word is present in the vocabulary, retry.")
    for i in range(length):

      text_idxs = [word2idx(word,w2v_model) for word in text_words]
      text_idxs = torch.LongTensor(text_idxs).view(1,len(text_idxs)).to(device)
      output, _ = net(text_idxs)
      output = nn.functional.softmax(output[:,-1,:])

      predicted_idx = np.random.choice( np.arange( 0, output.shape[1] ),
                                        1,
                                        p=output.view(output.shape[1]).detach().cpu().numpy() )[0]
      predicted_word = idx2word(predicted_idx,w2v_model)
      text_words.append(predicted_word)



    #### Print post-processing
    capitalize_words = ['lannister', 'stark', 'lord', 'ser', 'tyrion', 'jon', 'john snow', 'daenerys', 'targaryen', 'cersei', 'jaime', 'arya', 'sansa', 'bran', 'rikkon', 'joffrey',
                    'khal', 'drogo', 'gregor', 'clegane', 'kings landing', 'winterfell', 'the mountain', 'the hound', 'ramsay', 'bolton', 'melisandre', 'shae', 'tyrell',
                   'margaery', 'sandor', 'hodor', 'ygritte', 'brienne', 'tarth', 'petyr', 'baelish', 'eddard', 'greyjoy', 'theon', 'gendry', 'baratheon', 'baraTheon',
                   'varys', 'stannis', 'bronn', 'jorah', 'mormont', 'martell', 'oberyn', 'catelyn', 'robb', 'loras', 'missandei', 'tommen', 'robert', 'lady', 'donella', 'redwyne'
                   'myrcella', 'samwell', 'tarly', 'grey worm', 'podrick', 'osha', 'davos', 'seaworth', 'jared', 'jeyne poole', 'rickard', 'yoren', 'meryn', 'trant', 'king', 'queen',
                   'aemon']

    punc_list = ['.',',','?','!','"',';']

    final_text_words = []
    for word in text_words:
      if word in capitalize_words:
        word = word.title()
        final_text_words.append(word)
      else:
        final_text_words.append(word)

    final_string = ''
    for i,w in enumerate(final_text_words):
      if i!=0 and (final_text_words[i-1]=='.' or final_text_words[i-1]=='!' or final_text_words[i-1]=='?'):
        w = w.title()
      if w in punc_list:
        final_string = final_string + w
      else:
        final_string = final_string + ' ' + w
    print('\n\n GENERATED SEQUENCE')
    print('\n\n')
    print(final_string)
    print('\n\n')
