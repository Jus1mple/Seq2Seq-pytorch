# -*- coding:utf-8 -*-
# Trainer Object
# author: Matthew



import math
from config import MODEL_SAVE_PATH, RNN_CONFIG, RNN_MODEL_SAVE_PATH, SEQ2SEQ_MODEL_SAVE_PATH, AdditiveAttention_CONFIG, DotProductionAttention_CONFIG, Seq2Seq_CONFIG
from dataloader import SeqDataLoader
from model import RNN_Model, Seq2Seq_Model
from vocab import myVocab
from timer import Timer
from accumulator import Accumulator
import torch
import torch.nn as nn
import torch.optim as optim
from utils import MaskedSoftmaxCELoss, get_device, masked_softmax, try_gpu



def grad_clipping(net, theta):
    """Clip the gradient."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

class RNN_Trainer:
    """Trainer for RNN models"""
    def __init__(self, config : RNN_CONFIG):
        self.name = config.name
        self.lr = config.lr
        self.device_name = config.device
        self.device = try_gpu() if config.device == "GPU" else torch.cuda.device("cpu")
        self.num_epoches = config.num_epoches
        self.train_name = config.train_name
    
    def train(self, model : RNN_Model, train_iter : SeqDataLoader, vocab : myVocab, num_epoches = -1, lr = 0.0, device = None, saving = True):
        """Train a Model"""
        import os
        if num_epoches == -1:
            num_epoches = self.num_epoches
        if lr == 0.0:
            lr = self.lr
        if device is None:
            device = self.device
        loss = nn.CrossEntropyLoss()
        updater = optim.SGD(model.parameters(), self.lr)
        for epoch in range(self.num_epoches):
            ppl, spped, today = self._train_epoch(model, train_iter, loss, updater, self.device)
            # if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}; perplexity {ppl:.6f}, {spped:.1f} tokens / sec on {str(self.device)}")
            if saving:
                torch.save(model.state_dict(), os.path.join(RNN_MODEL_SAVE_PATH, f"rnn_train_{self.train_name}_on_{self.device_name}_epoch{epoch + 1} _{today}.pt"))


    def _train_epoch(self, model : RNN_Model, train_iter : SeqDataLoader, loss, updater, device):
        state, timer = None, Timer()
        metric = Accumulator(2) # sum of training loss, no. of tokens
        cnt = 0
        for X, Y in train_iter:
            cnt += 1
            if cnt % 1000 == 0:
                print("training X count: ", cnt)
            if state is None:
                state = model.init_state(device = device, batch_size = X.shape[0])
            else:
                if isinstance(model, nn.Module) and not isinstance(state, tuple):
                    # `state` is a tensor for `nn.GRU`
                    state.detach_()
                else:
                    # `state` is tuple of tensor for `nn.LSTM`
                    for s in state:
                        s.detach_()
            y = Y.reshape(-1).T
            # print(X.shape)
            X = X.to(device)
            y = y.to(device)
            y_hat, state = model(X, state)
            # print("y_hat.shape: ", y_hat.shape)
            # print("y.shape: ", y.shape)
            l = loss(y_hat, y.long()).mean()
            if isinstance(updater, optim.Optimizer):
                updater.zero_grad()
                l.backward()
                grad_clipping(model, 1)
                updater.step()
            else:
                l.backward()
                grad_clipping(model, 1)
                updater(batch_size = 1)
            metric.add(l * y.numel(), y.numel())
        return math.exp(metric[0] / metric[1]), metric[1] / timer.stop(), timer.today()


class Seq2Seq_Trainer:
    """Sequence 2 Sequence Model Trainer"""
    def __init__(self, config : Seq2Seq_CONFIG):
        self.num_epoches = config.num_epoches
        self.lr = config.lr
        self.device_name = config.device
        self.device = get_device(config.device)
        self.train_name = config.train_name
        self.saving = config.saving
        pass
    
    def train(self, model : Seq2Seq_Model, train_iter, num_epoches = -1, lr = -1.0, device = None, saving = True):
        """Train Seq2Seq Model"""
        import os
        def xavier_init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
            elif type(m) == nn.GRU or type(m) == nn.LSTM:
                for param in m._flag_weights_names:
                    if "weight" in param:
                        nn.init.xavier_uniform_(m._parameters[param])
        num_epoches = num_epoches if num_epoches > 0 else self.num_epoches
        lr = lr if lr >= 0.0 else self.lr
        device = device if device is not None else self.device
        # loss = MaskedSoftmaxCELoss()
        model.apply(xavier_init_weights)
        model.to(device)
        loss = nn.CrossEntropyLoss()
        updater = optim.Adam(model.parameters(), lr = self.lr)
        
        for epoch in range(num_epoches):
            
            ppl, spped, train_loss, today = self._train_epoch(model, train_iter, loss, updater, device)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch{epoch + 1}: perplexity: {ppl:.3f}; train Loss: {train_loss:.3f}; spped: {spped:.3f} tokens / sec on {str(self.device)}; date: {today}")
                if saving:
                    torch.save(model.state_dict(), os.path.join(SEQ2SEQ_MODEL_SAVE_PATH, f"seq2seq_train_{self.train_name}_on_{self.device_name}_epoch{epoch + 1} _{today}.pt"))
    
    def _train_epoch(self, model, train_iter, loss, updater, device):
        """Training Step for each epoch"""
        timer = Timer()
        metric = Accumulator(2)
        train_cnt = 0
        epoch_loss = 0
        for X, Y in train_iter:
            train_cnt += 1
            if train_cnt % 100000 == 0:
                print(f"Training data number: {train_cnt}...")
            # X.shape = [batch_size, seq_len, num_inputs]
            X = X.to(device)
            # Y.shape = [batch_size, tgt_len, output_size]
            Y = Y.to(device)
            # zero grad
            updater.zero_grad()
            
            # get prediction
            # y_hat.shape = [tgt_len, batch_size, output_size]
            y_hat = model(X, Y)
            # calc loss
            l = loss(y_hat, Y.tranpose(0, 1)).sum()
            l.backward()
            grad_clipping(model, 1)
            # update model parameters
            updater.step()
            epoch_loss += l.item()
            metric.add(l * Y.numel(), Y.numel())
        return math.exp(metric[0] / metric[1]), metric[1] / timer.stop(), epoch_loss / len(train_iter), timer.today()
    
    def evaluate(self, model, val_iter, loss, device = None):
        """Evaluate Model Effect"""
        device = self.device if device is None else device
        model.eval()
        epoch_loss = 0
        timer = Timer()
        metric = Accumulator(2)
        with torch.no_grad():
            for i, (X, Y) in enumerate(val_iter):
                X = X.to(device)
                Y = Y.to(device)
                
                # turn off teacher_forcing_ratio
                y_hat = model(X, Y, teacher_forcing_ratio = 0)
                l = loss(y_hat, Y.transpose(0, 1)).sum()
                epoch_loss += l.item()
                metric.add(l * Y.numel(), Y.numel())
        return math.exp(metric[0] / metric[1]), metric[1] / timer.stop(), epoch_loss / len(val_iter), timer.today()

