#!/usr/bin/env python3

#  Feedforward Neural Network
#
import json
import math
import os
from pathlib import Path
import random
import time
from tqdm.notebook import tqdm, trange
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm.notebook import tqdm, trange


emotion_to_idx = {
    "anger": 0,
    "fear": 1,
    "joy": 2,
    "love": 3,
    "sadness": 4,
    "surprise": 5,
}
idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}
UNK = "<UNK>"


def fetch_data(train_data_path, val_data_path, test_data_path):
    """fetch_data retrieves the data from a json/csv and outputs the validation
    and training data

    :param train_data_path:
    :type train_data_path: str
    :return: Training, validation pair where the training is a list of document, label pairs
    :rtype: Tuple[
        List[Tuple[List[str], int]],
        List[Tuple[List[str], int]],
        List[List[str]]
    ]
    """
    with open(train_data_path) as training_f:
        training = training_f.read().split("\n")[1:-1]
    with open(val_data_path) as valid_f:
        validation = valid_f.read().split("\n")[1:-1]
    with open(test_data_path) as testing_f:
        testing = testing_f.read().split("\n")[1:-1]
	
    tra = []
    val = []
    test = []
    for elt in training:
        if elt == '':
            continue
        txt, emotion = elt.split(",")
        tra.append((txt.split(" "), emotion_to_idx[emotion]))
    for elt in validation:
        if elt == '':
            continue
        txt, emotion = elt.split(",")
        val.append((txt.split(" "), emotion_to_idx[emotion]))
    for elt in testing:
        if elt == '':
            continue
        txt = elt
        test.append(txt.split(" "))

    return tra, val, test


def make_vocab(data):
    """make_vocab creates a set of vocab words that the model knows

    :param data: The list of documents that is used to make the vocabulary
    :type data: List[str]
    :returns: A set of strings corresponding to the vocabulary
    :rtype: Set[str]
    """
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


def make_indices(vocab):
	"""make_indices creates a 1-1 mapping of word and indices for a vocab.

	:param vocab: The strings corresponding to the vocabulary in train data.
	:type vocab: Set[str]
	:returns: A tuple containing the vocab, word2index, and index2word.
		vocab is a set of strings in the vocabulary including <UNK>.
		word2index is a dictionary mapping tokens to its index (0, ..., V-1)
		index2word is a dictionary inverting the mapping of word2index
	:rtype: Tuple[
		Set[str],
		Dict[str, int],
		Dict[int, str],
	]
	"""
	vocab_list = sorted(vocab)
	vocab_list.append(UNK)
	word2index = {}
	index2word = {}
	for index, word in enumerate(vocab_list):
		word2index[word] = index 
		index2word[index] = word 
	vocab.add(UNK)
	return vocab, word2index, index2word 


class EmotionDataset(Dataset):
    """EmotionDataset is a torch dataset to interact with the emotion data.

    :param data: The vectorized dataset with input and expected output values
    :type data: List[Tuple[List[torch.Tensor], int]]
    """
    def __init__(self, data):
        self.X = torch.cat([X.unsqueeze(0) for X, _ in data])
        self.y = torch.LongTensor([y for _, y in data])
        self.len = len(data)
    
    def __len__(self):
        """__len__ returns the number of samples in the dataset.

        :returns: number of samples in dataset
        :rtype: int
        """
        return self.len
    
    def __getitem__(self, index):
        """__getitem__ returns the tensor, output pair for a given index

        :param index: index within dataset to return
        :type index: int
        :returns: A tuple (x, y) where x is model input and y is our label
        :rtype: Tuple[torch.Tensor, int]
        """
        return self.X[index], self.y[index]

def get_data_loaders(train, val, batch_size=16):
    """
    """
    dataset = EmotionDataset(train + val)

    train_indices = [i for i in range(len(train))]
    val_indices = [i for i in range(len(train), len(train) + len(val))]

    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=batch_size,  sampler=val_sampler)

    return train_loader, val_loader

train, val, test = fetch_data(train_path, val_path, test_path)

vocab = make_vocab(train)
vocab, word2index, index2word = make_indices(vocab)

# Lambda to switch to GPU if available
get_device = lambda : "cuda:0" if torch.cuda.is_available() else "cpu"

unk = '<UNK>'


def rnn_preprocessor(data, max_len, word2index, test=False):
    """rnn_preprocessing 
    """
    # Do some preprocessing similar to convert_to_vector_representation
    # For the RNN, remember that instead of a single vector per training
    # example, you will have a sequence of vectors where each vector
    # represents some information about a specific token.
    features = []
    if test:
      #vectorized_data = torch.zeros(len(data),len(word2index))
      
      for document in data:
        vectorized_data = torch.zeros(max_len).long()
        num = 0
        for word in document:
          if num < len(document):
          
            index = word2index.get(word, word2index[UNK])
            vectorized_data[max_len - len(document) + num] = index + 1
            num += 1 
        features.append(vectorized_data)
    else:
        
      for document, y in data:
        vectorized_data = torch.zeros(max_len).long()
        num = 0
         
        for word in document:
          if num < len(document):
            #vector = torch.zeros(len(word2index))
            index = word2index.get(word, word2index[UNK])
            vectorized_data[max_len - len(document) + num]= index + 1
            #vectorized_data.append((vector))
            num += 1
        features.append((vectorized_data, y))
    return features

#torch.cuda.clear_cache
f = lambda x:len(x[0])
max_len = max(list(map(f, train)))
#max_len = 20
train_rnn = rnn_preprocessor(train, max_len, word2index, test=False)
val_rnn = rnn_preprocessor(val, max_len, word2index, test=False)
test_rnn = rnn_preprocessor(test, max_len, word2index, True)

# Lambda to switch to GPU if available
get_device = lambda : "cuda:0" if torch.cuda.is_available() else "cpu"
class RNNModel(nn.Module):
	def __init__(self, word_size, output_dim, embedding_dim, h, n_layers):
	#def __init__(self, input_dim, h, output_dim): # Add relevant parameters
		super(RNNModel, self).__init__()
		self.output_dim=output_dim
		self.n_layers=n_layers
		self.h = h
        
		self.embedding=nn.Embedding(word_size, embedding_dim)

		self.rnn = nn.RNN(embedding_dim, h, n_layers, batch_first=True, nonlinearity='relu')
		self.out = nn.Linear(h, output_dim)
		# Ensure parameters are initialized to small values, see PyTorch documentation for guidance
		self.softmax = nn.LogSoftmax(dim=1)
		self.loss = nn.NLLLoss()

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)	

	def forward(self, inputs):
		# begin code
		h0 = torch.zeros(self.n_layers, inputs.size(0), self.h).to(get_device())
		embedd=self.embedding(inputs)
		out, hidden = self.rnn(embedd, h0) 
		output = self.out(out[:, -1,:]) #h -> 6
		predicted_vector = self.softmax(output) 
		# end code
		return predicted_vector,hidden
	'''
		def init_hidden(self, batch_size):
			weight = next(self.parameters()).data
			hidden = weight.new(self.n_layers, batch_size, self.h).zero_()
			return hidden
	'''
	def load_model(self, save_path):
		self.load_state_dict(torch.load(save_path))
	
	def save_model(self, save_path):
		torch.save(self.state_dict(), save_path)


def train_epoch(model, train_loader, optimizer):
	model.train()
	total_loss = []
	#h = model.init_hidden(batch_size).to(get_device())
	for (input_batch, expected_out) in tqdm(train_loader, leave=False, desc="Training Batches"):
		#h = h.data 
		optimizer.zero_grad() 
		output,h = model(input_batch.to(get_device()))
		loss = model.compute_Loss(output, expected_out.to(get_device()))
		loss.backward()
		optimizer.step()
		total_loss.append(loss)
		#h0 = h
	print(sum(total_loss))
	return 
def evaluation(model, val_loader, optimizer):
	model.eval()
	correct = 0
	total = 0
	for (input_batch, expected_out) in tqdm(val_loader, leave=False, desc="Validation Batches"):
		output,val_h = model(input_batch.to(get_device()))
		total += output.size()[0]
		_, predicted = torch.max(output, 1)
		correct += (expected_out== predicted.to("cpu") ).cpu().numpy().sum()
	accuracy =  correct / total
	#loss /= len(val_loader) # should use correct / total?
	# Print validation metrics
	print('Accuracy: {}'.format(accuracy))

def train_and_evaluate(number_of_epochs, model, train_loader, val_loader):
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	for epoch in trange(number_of_epochs, desc="Epochs"):
		train_epoch(model, train_loader, optimizer)
		evaluation(model, val_loader, optimizer)
	return


import torchtext
glove = torchtext.vocab.GloVe(name="6B", dim=50, max_vectors=10000) # use 10k most

def embedding_matrix(glove, word2index):
    embedding_dim = glove.vectors.shape[1]
    weight_matrix = torch.zeros(len(word2index)+1,embedding_dim)
    for i in word2index:
      if i in glove.stoi:
         weight_matrix[word2index[i]+1] = glove.vectors[glove.stoi[i]]
      else:
          unk_vec = torch.ones(1, embedding_dim)
          nn.init.xavier_uniform_(unk_vec)
          weight_matrix[word2index[UNK]+1] = unk_vec
      padding_vec = torch.ones(1, embedding_dim)
      nn.init.xavier_uniform_(padding_vec)
      weight_matrix[0] = padding_vec

    return weight_matrix


weight_matrix = embedding_matrix(glove, word2index)

class emRNN(nn.Module):
    def __init__(self, input_dim, h, output_dim,weight_matrix,n_layers):
        super(emRNN, self).__init__()
        self.output_dim=output_dim
        self.n_layers=n_layers
        self.h = h
        self.weight_matrix = weight_matrix
        embedding_dim = weight_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(weight_matrix)
        
        self.rnn = nn.RNN(embedding_dim, h, n_layers, batch_first=True)
        self.fc = nn.Linear(h, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
    
    def forward(self, inputs):
        h0 = torch.zeros(self.n_layers, inputs.size(0), self.h).to(get_device())
        embedd=self.embedding(inputs)
        out, hidden = self.rnn(embedd, h0) 
        output = self.fc(out[:, -1,:]) #h -> 6
        predicted_vector = self.softmax(output) 
        return predicted_vector,hidden

    def compute_Loss(self, predicted_vector, gold_label):
      return self.loss(predicted_vector, gold_label)	

    def load_model(self, save_path):
      self.load_state_dict(torch.load(save_path))
	
    def save_model(self, save_path):
      torch.save(self.state_dict(), save_path)


word_size = len(word2index)+1 
output_dim = len(emotion_to_idx)
embedding_dim = 500
h = 256
n_layers = 2
model = RNNModel(word_size, output_dim, embedding_dim, h, n_layers).to(get_device())
print(model)


# In[23]:


batch_size=20
train_loader, val_loader = get_data_loaders(train_rnn, val_rnn, batch_size=batch_size)
#train_and_evaluate(50, model, train_loader, val_loader)
#model.save_model(os.path.join(os.getcwd(), "drive", "My Drive", "Colab Notebooks", "rnn_2layer_gl.pth"))


