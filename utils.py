# -*- coding:utf-8 -*-
# utils
# author: Matthew
import collections
import os

from config import PASSWORD_DATASET_PATH, DataLoader_CONFIG, printable_list
from sympy import sequence
from vocab import START, PAD, END
import torch.nn as nn
import torch

def count_corpus(tokens):
    """Count token frequencies"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def tokenize(lines, level = "char"):
    """split text lines into char or word tokens"""
    if level == "word":
        return [line.split(' ') for line in lines]
    elif level == "char":
        return [list(line) for line in lines]
    else:
        print("ERROR: unknown token level: ", level)


def padding_lines(lines, bos = START, eos = END, pad = PAD, max_len = 31, add_pad = True):
    res = []
    for line in lines:
        # print(line)
        tmp = [bos] + line + [eos]
        if add_pad:
            if len(tmp) < max_len + 2:
                tmp_cnt = max_len + 2 - len(tmp)
                tmp += [pad] * tmp_cnt
        res.append(tmp)
    return res


def truncate_pad_line(line, max_length, padding_token):
    """Truncate OR Pad a line"""
    if len(line) > max_length:
        return line[: max_length]
    else:
        return line + [padding_token] * (max_length - len(line))


def load_password_dataset(filename, with_count = False, pw_idx = 0, freq_idx = 1, spliter = "\t"):
    """load passwords from data file.
    
    : `with_count`: file format whether is with-count. Default. False.\n
    : `pw_idx`: if file is with-count, then specify the password index.\n
    : `freq_idx`: if file is with-count, then specify the frequency index.\n
    : `spliter`: the spliter for each line in the file.\n
    
    Returns:
        dict, list: password dict, password list(shuffle)
    """
    import random
    file_path = os.path.join(PASSWORD_DATASET_PATH, filename + ".txt")
    if not os.path.exists(file_path):
        print("No file exists")
        return {}, []
    psw_dict = collections.defaultdict(int)
    psw_list = []
    with open(file_path, 'r', encoding = 'utf-8', errors = 'ignore') as fin:
        for line in fin:
            line = line.strip('\r\n')
            # print(line)
            if with_count is False:
                # print("False")
                if not isValid(line):
                    continue
                psw_dict[line] += 1
                psw_list.append(line)
            else:
                ll = line.split(spliter)
                psw, freq = ll[pw_idx], ll[freq_idx]
                if not isValid(psw):
                    continue
                psw_dict[psw] += int(freq)
                psw_list.extend([psw] * int(freq))
    random.shuffle(psw_list)
    return psw_dict, psw_list


def get_dataloader(config : DataLoader_CONFIG):
    """Return the iterator and the vocabulary of the `filename` dataset"""
    from dataloader import SeqDataLoader
    data_iter = SeqDataLoader(config)
    return data_iter, data_iter.vocab


# small function
def isValid(psw):
    """check if the password is valid or not"""
    if len(psw) < 6 or len(psw) > 31:
        # only accept password length in [6, 31]
        return False
    # print(len(get_printable()))
    for c in psw:
        if c not in printable_list:
            # print("False")
            return False
    return True


def try_gpu(i = 0):
    """Return gpu(i) if exists, else return cpu()"""
    import torch
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    else:
        return torch.device('cpu')


def try_all_gpus():
    """Return all available GPUs, or [cpu()], if GPU is not exist"""
    import torch
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device("cpu")]


def get_device(device_type = "GPU"):
    """get the `device_type` specified device(default is GPU)"""
    import torch
    if device_type is "GPU":
        return try_gpu()
    elif device_type is "CPU":
        return torch.device("cpu")
    else:
        return try_gpu()


def sequence_mask(X, valid_len, value = 0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype = torch.float32, device = X.device)[None, : ] < valid_len[ : , None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X` is a 3D tensor, `valid_lens` is a 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim = -1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elemetns with a very large negative value
        # whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value = -1e6)
        return nn.functional.softmax(X.reshape(shape), dim = -1)


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks."""
    def forward(self, pred, label, valid_len):
        """
            `pred` shape [batch_size, num_steps, voacb_size]
            `label` shape [batch_size, num_steps]
            `valid_len` shape [batch_size, ]
        """
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = "none"
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim = 1)
        return weighted_loss


def bleu(pred_seq, label_seq, k):
    """Compute the BLEU"""
    import math
    import collections
    pred_tokens, label_tokens = pred_seq.split(" "), label_seq.split(" ")
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i : i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i : i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i : i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads"""
    # Shape of X:
    # [batch_size, no. of queries or key-value pairs, num_hiddens]
    # Shape of output X:
    # [batch_size, no. of queries or key-value pairs, num_heads, num_hiddens / num_heads]
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # tranpose X
    # after that, X.shape = [batchsize, num_heads, no. of queries or key-value pairs, num_hiddens / num_heads]
    X = X.permute(0, 2, 1, 3)
    
    # reshape X
    # after that X.shape = [batch_size * num_heads, no. of queries or key-value pairs, num_hiddens / num_heads]
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(output, num_heads):
    """Reverse the operation of `transpose_qkv`"""
    output = output.reshape(-1, num_heads, output.shape[1], output.shape[2])
    output = output.permute(0, 2, 1, 3)
    # output shape is [batch_size, no. of queries or key-value pairs, num_hiddens]
    return output.reshape(output.shape[0], output.shape[1], -1)