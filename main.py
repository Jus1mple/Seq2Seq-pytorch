# -*- coding:utf-8 -*-
# Project Main Entrance 
# author: Matthew


from config import MODEL_SAVE_PATH, RNN_CONFIG, RNN_GUESS_CONFIG, DataLoader_CONFIG
from utils import get_dataloader, try_gpu
from guess import RNN_Guesser
from model import RNN_Model
from train import RNN_Trainer
import os
import torch

# 设置随机数种子，确保可以后面复现实验
torch.manual_seed(0)

def RNN_process():
    """process LSTM/GRU(RNN) Model training and predicting"""
    guess_config = RNN_GUESS_CONFIG(vocab)
    device = try_gpu()
    dataloader_config = DataLoader_CONFIG()
    train_iter, vocab = get_dataloader(dataloader_config)
    rnn_config = RNN_CONFIG(vocab)
    model = RNN_Model(config = rnn_config)
    def rnn_train(model, trainer, saving = True):
        model = model.to(device = device)
        trainer = RNN_Trainer(config = rnn_config)
        trainer.train(model = model, train_iter = train_iter, vocab = vocab, device = device, saving = saving)
    def rnn_guess(model_name):
        _model = RNN_Model(rnn_config)
        _model = _model.to(device)
        _model.load_state_dict(os.path.join(MODEL_SAVE_PATH, model_name))
        guesser = RNN_Guesser(config = guess_config)
        guesser.predict(model = _model, vocab = vocab, device = device)
    rnn_train(model)
    rnn_guess(model_name = f"rnn_train_{rnn_config.train_name}_on_{rnn_config.device}_epoch{rnn_config.num_epoches}_20220627.pt")


def Seq2Seq_process():
    """process Seq2Seq Model training and predicting"""


def main():
    """main function"""
    RNN_process()


if __name__ == "__main__":
    """program entrance"""
    main()