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


def convert_to_vector_representation(data, word2index, test=False):
	"""convert_to_vector_representation converts the list of strings into a vector

	:param data: The dataset to be converted into a vectorized format
	:type data: Union[
		List[Tuple[List[str], int]],
		List[str],
	]
	:param word2index: A mapping of word to index
	:type word2index: Dict[str, int]
	:returns: A list of vector representations of the input or pairs of vector
		representations with expected output
	:rtype: List[Tuple[torch.Tensor, int]] or List[torch.Tensor]
	"""
	if test:
		vectorized_data = []
		for document in data:
			vector = torch.zeros(len(word2index)) 
			for word in document:
				index = word2index.get(word, word2index[UNK])
				vector[index] += 1
			vectorized_data.append(vector)
	else:
		vectorized_data = []
		for document, y in data:
			vector = torch.zeros(len(word2index)) 
			for word in document:
				index = word2index.get(word, word2index[UNK])
				vector[index] += 1
			vectorized_data.append((vector, y))
	return vectorized_data


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
train_vectorized = convert_to_vector_representation(train, word2index)
val_vectorized = convert_to_vector_representation(val, word2index)
test_vectorized = convert_to_vector_representation(test, word2index, True)

train_loader, val_loader = get_data_loaders(train_vectorized, val_vectorized, batch_size=20)

print(len(vocab))
f = lambda x:len(x[0])
print(max(list(map(f, val))))
print(len(train_vectorized))
print(len(val_vectorized))
print(train_loader.dataset.X.shape)
print(val_loader.sampler)


# Lambda to switch to GPU if available
get_device = lambda : "cuda:0" if torch.cuda.is_available() else "cpu"

unk = '<UNK>'

class FFNN(nn.Module):
	def __init__(self, input_dim, h, output_dim):
		super(FFNN, self).__init__()
		self.h = h
		self.W1 = nn.Linear(input_dim, h)
		self.activation = nn.ReLU()
		self.W2 = nn.Linear(h, output_dim)
		self.softmax = nn.LogSoftmax(dim=1)
		self.loss = nn.NLLLoss()

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)

	def forward(self, input_vector):
		
		z1 = self.W1(input_vector)
		z1 = self.activation(z1)
		z2 = self.W2(z1) 
		predicted_vector = self.softmax(z2)
		return predicted_vector
	def load_model(self, save_path):
		self.load_state_dict(torch.load(save_path))
	
	def save_model(self, save_path):
		torch.save(self.state_dict(), save_path)


def train_epoch(model, train_loader, optimizer):
	model.train()
	total = 0
	loss = 0
	correct = 0
	for (input_batch, expected_out) in tqdm(train_loader, leave=False, desc="Training Batches"):
		optimizer.zero_grad() 
		output = model(input_batch.to(get_device())).cuda()
		# seems unnecessary 
		#total += output.size()[0]
		#_, predicted = torch.max(output, 1)
		#correct += (expected_out == predicted.to("cpu")).cpu().numpy().sum()

		loss = model.compute_Loss(output, expected_out.to(get_device()))
		loss.backward()
		optimizer.step()
	# Print accuracy

	return


def evaluation(model, val_loader, optimizer):
	model.eval()
	loss = 0
	correct = 0
	total = 0
	for (input_batch, expected_out) in tqdm(val_loader, leave=False, desc="Validation Batches"):
		output = model(input_batch.to(get_device()))
		total += output.size()[0]
		_, predicted = torch.max(output, 1)
		correct += (expected_out == predicted.to("cpu")).cpu().numpy().sum()

		loss += model.compute_Loss(output, expected_out.to(get_device()))
	loss /= len(val_loader)
	# Print validation metrics
	print(correct / total)

	pass

def train_and_evaluate(number_of_epochs, model, train_loader, val_loader):
	optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
	for epoch in trange(number_of_epochs, desc="Epochs"):
		train_epoch(model, train_loader, optimizer)
		evaluation(model, val_loader, optimizer)
	return

h = 256
model = FFNN(len(vocab), h, len(emotion_to_idx))
model.to(get_device())

loaded_model = FFNN(len(vocab), h, len(emotion_to_idx)).to(get_device())
loaded_model.load_model(os.path.join(os.getcwd(), "drive", "My Drive", "Colab Notebooks", "ffnn_fixed3.pth"))
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
evaluation(loaded_model, val_loader, optimizer)



