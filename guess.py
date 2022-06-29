# -*- coding:utf-8 -*-
# Guesser Object
# author: Matthew



from config import END, RNN_GUESS_CONFIG, PAD, PASSWORD_DATASET_PATH, START, Seq2Seq_GUESS_CONFIG
from model import RNN_Model, Seq2Seq_Model
from vocab import myVocab
import torch
import torch.nn as nn
import collections
from utils import get_device, truncate_pad_line, try_gpu
import os
from queue import PriorityQueue


class Guesser:
    def __init__(self, name) -> None:
        self.name = name
    
    def predict(self):
        raise NotImplementedError


class RNN_Guesser(Guesser):
    """RNN model Guesser predictor"""
    def __init__(self, config : RNN_GUESS_CONFIG) -> None:
        super(RNN_Guesser, self).__init__(config.guesser_name)
        self.guessnumber = config.guessnumber
        self.prob_threshold = config.threshold
        self.min_length = config.min_length
        self.max_length = config.max_length
        self.vocab = config.vocab
        self.eos = config.endsymbol
        self.pad = config.pad
        self.bos = config.bos
        self.device = try_gpu() if config.device == "GPU" else torch.cuda.device("cpu")
        self.curr_guessnumber = 0
        self.guess_dict = collections.defaultdict(float)
        self.softmax = nn.Softmax(dim = 1)
        self.fout = open(os.path.join(PASSWORD_DATASET_PATH, "generate.txt"), 'w', encoding = 'utf-8', errors = 'ignore')
        
        
    def predict(self, model : RNN_Model, vocab : myVocab, device):
        """use RNN model to predict posible characters"""
        model.eval()
        prefix = PriorityQueue()
        prefix.put((-1.0, [self.bos]))
        while (not prefix.empty()) and self.curr_guessnumber < self.guessnumber:
            # do generate passwords while guessnumber is less than we need
            curr_prefix = prefix.get()
            # curr_prefix = curr_prefix[1]
            curr_prob = -curr_prefix[0]
            # print(curr_prob)
            outputs = curr_prefix[1].copy()
            if len(outputs) > self.max_length + 1:
                continue
            state = model.init_state(device = device, batch_size = 1)
            x = torch.tensor(outputs).reshape(1, -1).to(torch.float32) # x.shape = [1, len(outputs)]
            x = x.to(device = device)
            y, state = model(x, state)
            next_list = self.softmax(y)[0].tolist()
            next_dict = {i : pred_prob for i, pred_prob in enumerate(next_list)}
            # sorted all posible characters
            sorted_next_list = sorted(next_dict.items(), key = lambda x : x[1], reverse = True)
            for idx, pred_prob in sorted_next_list:
                tmp = outputs.copy()
                tmp.append(idx)
                if pred_prob * curr_prob > self.prob_threshold:
                    if idx == self.pad:
                        continue
                    if idx == self.bos:
                        continue
                    if idx == self.eos:
                        guess = "".join(vocab.to_tokens(tmp[1:-1]))
                        if len(guess) >= self.min_length and len(guess) <= self.max_length:
                            print(f"{guess}\t{pred_prob * curr_prob}", file = self.fout)
                            self.guess_dict[guess] = pred_prob * curr_prob
                            self.curr_guessnumber += 1
                            if self.curr_guessnumber % 1000 == 0:
                                print(self.curr_guessnumber)
                    else:
                        prefix.put((-curr_prob * pred_prob, tmp))
                else:
                    break
        self.fout.close()


class Seq2Seq_Guesser(Guesser):
    def __init__(self, config: Seq2Seq_GUESS_CONFIG):
        super(Seq2Seq_Guesser, self).__init__(config.guesser_name)
        self.vocab = config.vocab
        self.eos = config.eos
        self.bos = config.bos
        self.pad = config.pad
        self.guessnumber = config.guessnumber
        self.max_length = config.max_length
        self.min_length = config.min_length
        self.threshold = config.threshold
        self.device = get_device(config.device)
        self.device_name = config.device
    
    def predict(self, model : Seq2Seq_Model, vocab : myVocab):
        # TODO : 设想如何完成猜测？如何生成口令？
        pass
    
    def predict_translate(self, model : Seq2Seq_Model, src, src_vocab : myVocab, tgt_vocab : myVocab, device, num_steps, save_attention_weights = False):
        """predict Seq2Seq Model to translate language like FROM `English` TO `French`"""
        model.eval()
        # src_tokens type is list, len(src_tokens) == len(src.split(' )) + 2
        src_tokens = [self.bos] + src_vocab[src.split(' ')] + [self.eos] # no need for lower() src
        enc_valid_len = torch.tensor([len(src_tokens)], device = device) 
        src_tokens = truncate_pad_line(src_tokens, max_length = self.max_length + 2, padding_token = self.pad)
        # transfer list src into tensor X
        # add an extra dimension as `0`
        # add `batch_size` dimension
        enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype = torch.long, device = device), dim = 0) 
        # dec_state = (hidden, cell)
        enc_output, state = model.encoder(enc_X, enc_valid_len)
        
        # Add the `batch_size` dimension
        dec_X = torch.unsqueeze(torch.tensor([self.bos], dtype = torch.long, device = device), dim = 0)
        output_seq, attention_weight_seq = [], []
        for _ in range(num_steps):
            Y, state = model.decoder(dec_X, state) # decoder.forward(X, hidden, cell) after two params are state(LSTM as a tuple)
            # use the token with the hightest prediction likelihood as the input of the decoder at the next tiemstep
            dec_X = Y.argmax(dim = 2)
            pred = dec_X.squeeze(dim = 0).type(torch.int32).item()
            # Save attention weights
            if save_attention_weights:
                attention_weight_seq.append(model.decoder.attention_weights)
            # Once the end of sequence token is predicted, the generation of the output sequence is complete
            if pred == self.eos:
                break
            output_seq.append(pred)
        return " ".join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
        