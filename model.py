# -*- coding:utf-8 -*-
# Model Object
# author: Matthew

import math
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FFN_CONFIG, RNN_CONFIG, RNN_MODEL_NAME, AddNorm_CONFIG, AdditiveAttention_CONFIG, Decoder_CONFIG, DotProductionAttention_CONFIG, Encoder_CONFIG, EncoderBlock_CONFIG, MultiHeadAttention_CONFIG, PositionalEncoding_CONFIG, Seq2Seq_CONFIG, TransformerEncoder_CONFIG
from utils import get_device, masked_softmax, transpose_output, transpose_qkv, try_gpu
import random


class RNN_Model(nn.Module):
    """RNN Model"""
    def __init__(self, config:RNN_CONFIG) -> None:
        super(RNN_Model, self).__init__()
        self.name = config.name
        self.vocab_size = config.vocab_size
        self.num_inputs = config.num_inputs
        self.num_hiddens = config.num_hiddens
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.embedding = config.embedding
        self.embedding_dim = config.embedding_dim
        self.dropout = config.dropout
        
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx = config.vocab.pad)
            tmp_num_inputs = self.embedding_dim
        else:
            self.embedding_layer = None
            tmp_num_inputs = self.num_inputs
        
        if config.name == RNN_MODEL_NAME.LSTM:
            self.rnn = nn.LSTM(tmp_num_inputs, self.num_hiddens, self.num_layers, dropout = self.dropout, bidirectional = self.bidirectional)
        elif config.name == RNN_MODEL_NAME.GRU:
            self.rnn = nn.GRU(tmp_num_inputs, self.num_hiddens, self.num_layers, dropout = self.dropout, bidirectional = self.bidirectional)
        else:
            self.rnn = nn.LSTM(tmp_num_inputs, self.num_hiddens, self.num_layers, dropout = self.dropout, bidirectional = self.bidirectional)
        if not self.bidirectional:
            self.num_directions = 1
        else:
            self.num_directions = 2
        self.linear = nn.Linear(self.num_hiddens * self.num_directions, self.vocab_size)
    
    def forward(self, X, state):
        batch_size, seq_len = X.shape # origin X shape is : [batch_size, seq_len(=num_steps)]
        # print(X)
        if self.embedding:
            X = X.T # transpose X shape is [seq_len(=num_steps), batch_size]
            X = X.to(torch.float32)
            # embedding X.shape : [seq_len, batch_size, embedding_dim]
            X = self.embedding_layer(X) 
            last_dim = self.embedding_dim
        else:
            # ont_hot X.shape : [seq_len, batch_size, vocab_size]
            # X = X.T # transpose X shape is [seq_len(=num_steps), batch_size]
            # X = X.to(torch.float32)
            # print(X)
            X = X.long()
            X = F.one_hot(X, self.vocab_size)
            X = X.transpose(0, 1)
            last_dim = self.vocab_size
        Y, state = self.rnn(X.float(), state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state
    
    
    def init_state(self, device, batch_size = 1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return torch.zeros(
                (self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                device = device
            )
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (
                torch.zeros(
                    (self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device = device
                ),
                torch.zeros(
                    (self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device = device
                ),
            )


class BaseEncoder(nn.Module):
    """The base encoder interface"""
    def __init__(self, **kwargs) -> None:
        super(BaseEncoder, self).__init__(**kwargs)
    
    def forward(self, X, *args):
        raise NotImplementedError
    
    def init_state(self):
        raise NotImplementedError


class BaseDecoder(nn.Module):
    """The base decoder interface"""
    def __init__(self, **kwargs) -> None:
        super(BaseDecoder, self).__init__(**kwargs)

    def forward(self, X, hidden, cell, *args):
        raise NotImplementedError
    
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError


class Encoder(BaseEncoder):
    def __init__(self, config : Encoder_CONFIG, **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)
        self.num_hiddens = config.num_hiddens
        self.num_inputs = config.vocab_size
        self.num_layers = config.num_layers
        self.embedding_size = config.embedding_size
        self.dropout = config.dropout
        self.embedding_layer = nn.Embedding(self.num_inputs, self.embedding_size)
        # 下面是 Embedding Layer的函数实现版本 可以直接使用上面的
        self.dropout = nn.Dropout(self.dropout)
        self.relu = F.relu
        self.linear = nn.Linear(self.num_inputs, self.embedding_size)
        # 是否是双向LSTM
        self.bidirectional = config.bidirectional
        # 这里是Encoder，不需要最后的Linear，所以没啥用这个变量
        self.num_directions = 2 if self.bidirectional else 1
        # RNN Model 这里使用LSTM
        self.rnn = nn.LSTM(self.embedding_size, self.num_hiddens, self.num_layers, dropout = self.dropout, bidirectional = self.bidirectional)
    
    def forward(self, X, *args):
        # X.shape: [batch_size, seq_len, num_inputs]
        # do embedding opration
        # embedding X.shape : [batch_size, seq_len, embedding_size]
        # X = self.dropout(self.relu(self.linear(X)))
        X = self.embedding_layer(X)
        X = X.permute(1, 0, 2)
        X = X.to(torch.float32)
        # only use hidden and cell as context to feed into decoder
        # hidden = [num_layers * num_directions, batch_size, hidden_size]
        # cell = [num_layers * num_directions, batch_size, hidden_size]
        # state = (hidden, cell)
        output, state = self.rnn(X)
        return output, state

    def init_state(self, device, batch_size = 1):
        if not isinstance(self.rnn, nn.LSTM):
            # if rnn is not LSTM, state is a tensor whose shape is [num_directions * num_layers, batch_size, num_hiddens]
            return torch.zeros(
                (self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                device = device
            )
        else:
            # `nn.LSTM` take a tuple as hidden state, tuple(hidden, cell), their shape are the same [num_directions * num_layers, batch_size, num_hiddens]
            return (
                torch.zeros(
                    (self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                    device = device
                ),
                torch.zeros(
                    (self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                    device = device
                )
            )


class Decoder(BaseDecoder):
    def __init__(self, config : Decoder_CONFIG) -> None:
        super(Decoder, self).__init__()
        self.output_size = config.output_size # vocab_size
        self.num_layers = config.num_layers # num_layers
        self.num_hiddens = config.num_hiddens # num_hiddens
        self.dropout = config.dropout # model dropout default 0.5
        self.embedding_size = config.embedding_size # Embedding dimenstion
        self.dropout = nn.Dropout(self.dropout)
        self.relu = F.relu # relu function
        self.linear1 = nn.Linear(self.output_size, self.embedding_size) # linear input_shape = vocab_size, output_shape = embedding_size. That is, Embedding Layer combining with `relu` and `dropout`
        self.embedding_layer = nn.Embedding(self.output_size, self.embedding_size) # direct Embedding Layer. This is simple. ^_^
        self.bidirectional = config.bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn = nn.LSTM(self.embedding_size, self.num_hiddens, self.num_layers, dropout = self.dropout)
        self.linear2 = nn.Linear(self.num_hiddens * self.num_directions, self.output_size)
    
    def forward(self, X, state):
        """Decoder kernel

        Args:
            X (input batch data): size of X : [batch_size, num_inputs]
            hidden (hidden state): the Encoder hidden state
            cell (the output cell of encoder RNN): the output cell of encoder RNN.
            LSTM return its state as tuple (hidden, cell), when they init, they are the same.
        """
        # add one dimension to X, like [batch_size, num_inputs] -->(unsqueeze(0)) [seq_len(=1), batch_size, num_inputs]
        X = X.unsqueeze(0)
        # do embedding
        # X = self.dropout(self.relu(self.linear1(X))) # updat4 
        X = self.embedding_layer(X)
        X = X.to(torch.float32)
        
        # (hidden, cell) is LSTM output states
        # output = [seq_len(=1), batch_size, num_hiddens * num_directions]
        # hidden = [num_layers * num_directions(=1), batch_size, num_hiddens]
        # cell = [num_layers * num_directions(=1), batch_size, num_hiddens]
        output, state = self.rnn(X, state)
        
        # prediction.shape : [batch_size, output_size]
        prediction = self.linear2(output.squeeze(0))
        
        return prediction, state

    def init_state(self, enc_outputs, *args):
        """Init Decoder Model hidden state(hidden, cell)"""
        # enc_outputs = prediction, state(= hidden, cell)
        return enc_outputs[1]


class Seq2Seq_Model(nn.Module):
    def __init__(self, encoder : Encoder, decoder : Decoder, config : Seq2Seq_CONFIG) -> None:
        super(Seq2Seq_Model, self).__init__()
        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.device = get_device(config.device)
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.num_hiddens == decoder.num_hiddens, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"
    
    def forward(self, X, Y, teacher_forcing_ratio = 0.5):
        # X.shape is [batch_size, seq_len, num_inputs]
        # Y.shape is [bathch_size, tgt_len, output_size]
        X = X.permute(1, 0, 2)
        Y = Y.permute(1, 0, 2)
        # tranpose X.shape = [observed_seq_len, batch_size, num_inputs]
        batch_size = X.shape[1]
        # tranpose Y.shape = [target_len, batch_size, output_size]
        target_len = Y.shape[0]
        
        # store decoder outputs of each time step
        outputs = torch.zeros(Y.shape).to(self.device)
        enc_state = self.encoder.init_state(device = self.device, batch_size = batch_size) # !! deprecated
        # last hidden, cell state as the decoder init-state
        enc_output, state = self.encoder(X)
        
        # first input to decoder is last coordinate of X, this input's shape is : [1, batch_size, num_inputs]
        decoder_input = X[-1, :, :]
        
        for i in range(target_len):
            # run decoder for time step(time_steps == target_len)
            output, state = self.decoder(decoder_input, state)
            
            # place predictions in a tensor
            # holder each predicton for each timestep
            # output shape is : [batch_size, output_size]
            outputs[i] = output 
            
            # decide whether there are going to teacher forcing or not
            teacher_forcing = random.random() < teacher_forcing_ratio
            
            # output is the shape as input [batch_size, output_size]
            # here is how the teacher_forcing work
            decoder_input = Y[i] if teacher_forcing else output
        # outputs shape is : [target_len, batch_size, output_size(=vocab_size)]
        return outputs



class AdditiveAttention(nn.Module):
    """Additive attention"""
    def __init__(self, config : AdditiveAttention_CONFIG,  **kwargs) -> None:
        super(AdditiveAttention, self).__init__(**kwargs)
        self.key_size = config.key_size
        self.query_size = config.query_size
        self.num_hiddens = config.num_hiddens
        self.dropout_val = config.dropout
        self.W_k = nn.Linear(self.key_size, self.num_hiddens, bias = False)
        self.W_q = nn.Linear(self.query_size, self.num_hiddens, bias = False)
        self.W_v = nn.Linear(self.num_hiddens, 1, biase = False)
        self.dropout = nn.Dropout(self.dropout_val)
    
    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # torch.bmm() 计算两个张量的乘法，torch.bmm(a, b), a.shape = [b, h, w], b.shape = [b, w, m]; 也就是说
        # 两个张量的一维是相等的，然后第一个张量的第三个维度大小应该与第二个张量的第二个维度大小相等
        # 得到的结果就应该是 [b, h, m]
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """Scaled dot product attention"""
    def __init__(self, config : DotProductionAttention_CONFIG, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dropout_val = config.dropout
        self.dropout = nn.Dropout(self.dropout_val)
    
    def forward(self, queries, keys, values, valid_lens):
        # 根据论文Attention is All you need中提到的，这里应该是使用keys矩阵的维度
        d_k = keys.shape[-1]
        scores = torch.bmm(queries, keys.tranpose(1, 2)) / math.sqrt(d_k)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class AttentionDecoder(BaseDecoder):
    """The base Attention-based Decoder interface"""
    def __init__(self, **kwargs) -> None:
        super(AttentionDecoder, self).__init__(**kwargs)
    
    @property
    def attention_weights(self):
        raise NotImplementedError


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self, config : MultiHeadAttention_CONFIG, **kwargs) -> None:
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.key_size = config.key_size
        self.query_size = config.query_size
        self.value_size = config.value_size
        self.bias = config.bias
        self.cropout = config.dropout
        self.num_hiddens = config.num_hiddens
        self.num_heads = config.num_heads
        self.attention = DotProductAttention(config.get_dotproductattention_config())
        self.W_q = nn.Linear(self.query_size, self.num_hiddens, bias = self.bias)
        self.W_k = nn.Linear(self.key_size, self.num_hiddens, bias = self.bias)
        self.W_v = nn.Linear(self.value_size, self.num_hiddens, bias = self.bias)
        self.W_o = nn.Linear(self.num_hiddens, self.num_hiddens, bias = self.bias) # full connected Layer
    
    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, values:
        # [batch_size, no. of queries or key-value pairs, num_hiddens]
        # Shape of valid_lens
        # [batch_size, ] or [batch_size, no. of queries]
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats = self.num_heads, dim = 0)
        # output.shape = [batch_size * num_heads, no. of queries or key-value paris, num_hiddens / num_heads]
        output = self.attention(queries, keys, values, valid_lens)
        
        output_concat = transpose_output(output, self.num_heads)
        
        # return shape is [batch_size, no. of queries or key-value pairs, num_hiddens]
        return self.W_o(output_concat)


class PositionalEncoding(nn.Module):
    """positional Encoding"""
    def __init__(self, config : PositionalEncoding_CONFIG, **kwargs) -> None:
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = config.max_len
        self.dropout_ = config.dropout_
        self.num_hiddens = config.num_hiddens
        self.dropout = nn.Dropout(self.dropout_)
        self.P = torch.zeros((1, self.max_len, self.num_hiddens))
        X = torch.arange(
            self.max_len, dtype = torch.float32).reshape(-1, 1) / \
                torch.pow(10000, torch.arange(0, self.num_hiddens, 2, dtype = torch.float32) / \
                        self.num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class PositionWiseFFN(nn.Module):
    """PositionWise feed-forward network"""
    def __init__(self, config : FFN_CONFIG, **kwargs) -> None:
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.ffn_num_inputs = config.num_inputs
        self.ffn_num_hiddens = config.num_hiddens
        self.ffn_num_outputs = config.num_outputs
        self.dense1 = nn.Linear(self.ffn_num_inputs, self.ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(self.ffn_num_hiddens, self.ffn_num_outputs)
    
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """Residual connection followed by layer normalization"""
    def __init__(self, config : AddNorm_CONFIG, **kwargs) -> None:
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.normalized_shape)
    
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, config : EncoderBlock_CONFIG, **kwargs) -> None:
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(config.get_multiattention_config())
        self.addnorm1 = AddNorm(config.get_addnorm_config())
        self.ffn = PositionWiseFFN(config.get_ffn_config())
        self.addnorm2 = AddNorm(config.get_addnorm_config())
    
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, config : TransformerEncoder_CONFIG, **kwargs):
        self.num_hiddens = config.num_hiddens
        self.embedding = nn.Embedding(config.vocab_size, config.num_hiddens)
        self.pos_encoding = PositionalEncoding(config.get_positionalencoding_config())
        self.blks = nn.Sequential()
        for i in range(config.num_layers):
            self.blks.add_module(
                "block" + str(i), 
                EncoderBlock(config.get_encoderblock_config())
            )
    
    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X