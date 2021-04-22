import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from ntc.trainer import Trainer
from ntc.data_loader import DataLoader

from ntc.model.rnn import RNNClassifier
from ntc.model.cnn import CNNClassifier

def define_argparser():

    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)

    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--min_vacab_freq', type=int, default=5)
    p.add_argument('--max_vocab_size', type=int, default=999999)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--word_vec_size',type=int, default=256)
    p.add_argument('--dropout', type=int, default=.3)

    p.add_argument('--max_length',type=int, default=256)
    
    p.add_argument('--rnn', action='store_true')
    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--n_layers',type=int, default=4)

    p.add_argument('--cnn',action='store_true')
    p.add_argument('--use_batch_norm',action='store_true')
    p.add_argument('--window_size', type=int, nargs='*', default=['3,4,5'])
    p.add_argument('--n_filters', type=int, nargs='*',default=[100,100,100])

    config = p.parse_args()

    return config

def main(config):

    loaders = DataLoader(
        train_fn=config.train_fn,
        batch_size=config.batch_size,
        min_freq=config.min_vacab_freq,
        max_vocab= config.max_vocab_size,
        device=config.gpu_id,
    )

    print(
        '|train| = ',len(loaders.train_loader.dataset),
        '|valid| = ',len(loaders.vaild_loader.dataset),
    )

    vocab_size = len(loaders.text.vocab)
    n_classes = len(loaders.label.vocab)
    print('|vocab| =', vocab_size, '|classes| =',n_classes)

    if config.rnn is False and config.cnn is False:
        raise Exception('You need to specify an architecture to train. (--rnn or cnn')

    if config.rnn:
        model = RNNClassifier(
            input_size=vocab_size,
            word_vec_size = config.word_vec_size,
            hidden_size =config.hidden_size,
            n_classes=n_classes,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        )
        optimizer = optim.Adam(model.parameters())
        crit= nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        cnn_trainer= Trainer(config)
        cnn_model = cnn_trainer.train(
            model,
            crit,
            optimizer,
            loader.train_loader,
            loader.valid_loader,
        )


    if config.cnn:
        model = CNNClassifier(
            input_size =vocab_size,
            word_vec_size= config.word_vec_size,
            hidden_size = config.hidden_size,
            n_classes = n_classes,
            use_batch_norm= config.use_batch_norm
        )