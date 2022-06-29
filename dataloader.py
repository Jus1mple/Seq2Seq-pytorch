# -*- coding:utf-8 -*-
# DataLoader
# author: Matthew

import random
import torch
from torch.utils import data
from config import DataLoader_CONFIG
from utils import load_password_dataset, tokenize, padding_lines
from vocab import myVocab
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence
from torch.utils.data.dataloader import DataLoader




def seq_data_iter_random(corpus, batch_size, num_steps, padding_value = 2):
    """Generate a mini_batch of subsequences using random sampling"""
    offset = random.randint(0, num_steps - 1) # offset in [0, num_steps - 1]
    corpus = corpus[offset : ]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices) # shuffle indices list
    
    def data(pos):
        return corpus[pos : pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i : i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps, padding_value = 2):
    """Generate a minibatch of subsequences using sequential partitioning"""
    # offset = random.randint(0, num_steps - 1)
    offset = 0
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset : offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1 : offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i : i + num_steps]
        Y = Ys[:, i : i + num_steps]
        # print(X)
        yield X, Y


def seq_data_single_psw_sequential_with_pad_sequence(padding_lines, batch_size = 1, num_steps = -1, padding_value = 2):
    """Generate a minibatch of whole passwords sequential"""
    num_tokens = ((len(padding_lines) - 1) // batch_size) * batch_size
    Xs = [torch.tensor(line) for line in padding_lines[: num_tokens]]
    Ys = [torch.tensor(line) for line in padding_lines[1 : num_tokens + 1]]
    num_batches = num_tokens // batch_size
    # print(Xs.shape)
    for i in range(0, num_batches * batch_size, batch_size):
        X = pad_sequence(Xs[i : i + batch_size], padding_value = padding_value).transpose(0, 1)
        Y = pad_sequence(Ys[i : i + batch_size], padding_value = padding_value).transpose(0, 1)
        yield X, Y


def seq_data_single_psw_sequential(padding_lines, batch_size = 1, num_steps = -1, padding_value = 2):
    """Generate a minibatch of whole passwords sequential"""
    num_tokens = ((len(padding_lines) - 1) // batch_size) * batch_size
    Xs = torch.tensor(padding_lines[ : num_tokens])
    Ys = torch.tensor(padding_lines[1 : 1 + num_tokens])
    num_batches = num_tokens // batch_size
    # print(Xs.shape)
    for i in range(0, num_batches * batch_size, batch_size):
        X = Xs[i : i + batch_size, : ]
        Y = Ys[i : i + batch_size, : ]
        yield X, Y



def load_corpus_from_dataset(filename, max_tokens = -1, flatten = True, printable_list = None):
    """Return token indices and the Vocabulary of the password dataset"""
    
    _, lines = load_password_dataset(filename, with_count = False)
    tokens = tokenize(lines, level = "char")
    # flattern
    if flatten:
        tokens = padding_lines(lines = tokens, add_pad = True) # add <bos> <eos> and padding(<pad>)
        vocab = myVocab(tokens = tokens, reserved_tokens = printable_list)
        corpus = [vocab[token] for line in tokens for token in line]
    else:
        tokens = padding_lines(lines = tokens, add_pad = True)
        vocab = myVocab(tokens = tokens, reserved_tokens = printable_list)
        corpus = [vocab[line] for line in tokens]
    # print(len(corpus))
    if max_tokens > 0:
        # limited token counts
        corpus = corpus[:max_tokens]
    return corpus, vocab



class SeqDataLoader:
    """An iterator to load sequence data."""
    def __init__(self, config:DataLoader_CONFIG):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.use_random_iter = config.use_random_iter
        self.filename = config.filename
        self.max_tokens = config.max_tokens
        self.printable_list = config.printable_list
        self.padding_value = config.padding_value
        
        if self.use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        if self.num_steps == -1:
            self.data_iter_fn = seq_data_single_psw_sequential
        flatten = True if self.num_steps > 0 else False
        self.corpus, self.vocab = load_corpus_from_dataset(self.filename, self.max_tokens, flatten, self.printable_list)
        self.padding_value = self.vocab[self.padding_value]
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps, self.padding_value)


def pretranspose(X, Y):
    """transpose X, Y tensor (0, 1)"""
    return X.transpose(0, 1), Y.transpose(0, 1)



class WarppedDataLoader:
    """Generate WrappedDataLoader to preprocess data"""
    def __init__(self, dataloader : DataLoader, func):
        self.dataloader = dataloader
        self.func = func
    
    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        data_iter = iter(self.dataloader)
        for data in data_iter:
            yield self.func(data)
    