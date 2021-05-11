import sys
import argparse

import torch
import torch.nn as nn
from torchtext import data

from ntc.model.rnn import RNNClassifier
from ntc.model.cnn import CNNClassifier

def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)
    p.add_argument('--max_length',type=int ,default=256)

    p.add_argument('--drop_rnn', action='store_true')
    p.add_argument('--drop_cnn', action='store_true')

    config = p.parse_args()

    return config

def read_text(max_length=256):

    lines =[]

    for line in sys.stdin:
        if line.strip() != '':
            lines += [lines.strip().split(' ')[:max_length]]
        
    return lines

def define_field():
    return (
        data.Field(
            use_vocab=True,
            batch_first=True,
            include_lengths=False,
        ),
        data.Field(
            sequential=False,
            use_vocab=True,
            unk_token=None,
        )
    )


def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id <0 else 'cuda:%d' % config.gpu_id
    )

    train_config = save_data['config']
    rnn_best = saved_data['rnn']
    cnn_best = saved_data['cnn']
    vocab = saved_data['vocab']
    classed = saved_data['classes']

    vocab_size =len(vocab)
    