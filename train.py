import os
import argparse
import json
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trainner import Trainer
from model import Model
from dataset import Dataset, Tokenizer, load_tokenizer
from scheduler import CosineWithRestarts


parser = argparse.ArgumentParser()
parser.add_argument('--src_vocab', required=True)
parser.add_argument('--tgt_vocab', required=True)
parser.add_argument('--train_file', required=True)
parser.add_argument('--test_file', required=True)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--num_epoch', default=10, type=int)
parser.add_argument('--device', default='cpu')
parser.add_argument('--log_file', default='log.txt')
parser.add_argument('--model_config')

if __name__ == "__main__":
    args = parser.parse_args()

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=logging.INFO)

    print("Load vocab")
    tokenizer = load_tokenizer(args.src_vocab, args.tgt_vocab)

    print("Prepare data")
    train_ds = Dataset(args.train_file, tokenizer)
    test_ds = Dataset(args.test_file, tokenizer)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size)

    print("Init model")
    src_vocab_len = len(tokenizer.src_stoi)
    tgt_vocab_len = len(tokenizer.tgt_stoi)

    if args.model_config:
        with open(args.model_config) as f:
            config = json.load(f)
    else:
        config = {}

    model = Model(src_vocab_len, tgt_vocab_len, **config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    sched = CosineWithRestarts(optimizer, T_max=len(train_dl))
    trainner = Trainer(model, optimizer, train_dl, test_dl, device=args.device, scheduler=sched)

    print("Start training")
    trainner.train(args.num_epoch)
