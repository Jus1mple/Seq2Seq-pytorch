# -*- coding:utf-8 -*-
# test file
# author: Matthew

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence
from timer import Timer

from vocab import myVocab

# from vocab import Vocab

def test_reshape():
    """test reshape"""
    t1 = torch.tensor(
        [
            [1, 2, 3, 4], 
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    )
    print(t1.shape)
    print(t1.reshape(2, -1).shape)


def test_utils():
    """test utils"""
    from utils import load_password_dataset, padding_lines, tokenize
    _, psw_list = load_password_dataset("test.txt")
    # print(psw_list)
    lines = tokenize(psw_list, level = "char")
    tokens = padding_lines(lines)
    print(tokens)

def test_embedding():
    """test nn.Embedding"""
    from utils import get_dataloader
    from torch.nn.functional import one_hot
    vocab_size = 95
    data_iter, vocab = get_dataloader(filename = "test.txt", batch_size = 5, num_steps = -1, use_random_iter = False, max_tokens = 10000)
    # print(vocab)
    # print(data_iter)
    embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = 20, padding_idx = vocab.pad)
    # print("..")
    cnt = 0
    for X, y in data_iter:
        # print(X)
        cnt += 1
        # print(X.shape)
        # print(y.T.reshape(-1).shape)
        # x1 = embedding(X)
        # print(x1.transpose(0, 1))
    print(cnt)


def test_softmax():
    """test nn.Softmax"""
    softmax = nn.Softmax(dim = 1)
    y = torch.rand(size=[15,95])

    #y的size是2,2,3。可以看成有两张表，每张表2行3列
    net_1 = nn.Softmax(dim=0)
    net_2 = nn.Softmax(dim=1)
    # net_3 = nn.Softmax(dim=2)
    y_d1 = net_1(y)
    y_d2 = net_2(y)
    print('dim=0的结果是：\n',y_d1,"\n")
    print('dim=1的结果是：\n',y_d2,"\n")
    # print('dim=2的结果是：\n',net_3(y),"\n")
    print(y_d1.shape)
    print(y_d2.shape)




def test_numel():
    """test numel"""
    y = torch.tensor([1,2,3])
    print(y)
    print(y.numel())


def test_pad_sequence():
    """test pad_sequence"""
    t1 = torch.rand(10)
    print(t1)
    t2 = torch.rand(9)
    print(t2)
    t3 = torch.rand(8)
    print(t3)
    print(pad_sequence([t1, t2, t3]))
    l1 = [1, 2, 3, 4, 5, 6]
    tl1 = torch.tensor(l1)
    l2 = [1, 4, 5]
    tl2 = torch.tensor(l2)
    l3 = [4, 5, 6, 7]
    tl3 = torch.tensor(l3)
    # print(tl1.shape)
    print(pad_sequence([tl1, tl2, tl3], padding_value = 2).transpose(0, 1))


def test_list_printable_list():
    """test printable"""
    from config import printable_list
    pl = printable_list
    print(list(pl))


def test_myVocab():
    """test myVocab"""
    
    print(myVocab.pad)
    vocab = myVocab()
    print(vocab.pad)


def test_Timer():
    """test Timer class"""
    timer = Timer()
    print(timer.today())


    
if __name__ == "__main__":
    """main entrance"""
    # test_reshape()
    # test_embedding()
    # test_numel()
    # test_softmax()
    # test_pad_sequence()
    # test_list_printable_list()
    # test_myVocab()
    test_Timer()