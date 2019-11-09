import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trainner import Trainer
from model import Model
from dataset import Dataset, Tokenizer, load_tokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--src_vocab', required=True)
parser.add_argument('--tgt_vocab', required=True)
parser.add_argument('--train_file', required=True)
parser.add_argument('--test_file', required=True)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--num_epoch', default=10, type=int)
parser.add_argument('--device', default='cpu')

if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.src_vocab, args.tgt_vocab)

    train_ds = Dataset(args.train_file, tokenizer)
    test_ds = Dataset(args.test_file, tokenizer)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size)

    src_vocab_len = len(tokenizer.src_stoi)
    tgt_vocab_len = len(tokenizer.tgt_stoi)

    model = Model(src_vocab_len, tgt_vocab_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainner = Trainer(model, optimizer, train_dl, test_dl, device=args.device)

    trainner.train(args.num_epoch)
