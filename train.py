import os
import argparse
import json

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
parser.add_argument('--model_config')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--num_epoch', default=10, type=int)
parser.add_argument('--device', default='cpu')
parser.add_argument('--log_dir', default='logs')
parser.add_argument('--weight_dir', default='weight')

if __name__ == "__main__":
    args = parser.parse_args()

    print("Load vocab")
    tokenizer = load_tokenizer(args.src_vocab, args.tgt_vocab)

    print("Prepare data")
    train_ds = Dataset(args.train_file, tokenizer)
    test_ds = Dataset(args.test_file, tokenizer)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size)

    print("Init model")
    vocab_config = {
        'src_vocab_len': len(tokenizer.src_stoi),
        'tgt_vocab_len': len(tokenizer.tgt_stoi),
        'src_padding_value': tokenizer.src_stoi['[pad]'],
        'tgt_padding_value': tokenizer.tgt_stoi['[pad]'],
        'src_mask_value': tokenizer.src_stoi['[mask]'],
        'tgt_mask_value': tokenizer.tgt_stoi['[mask]']
    }

    if args.model_config:
        with open(args.model_config) as f:
            config = json.load(f)
    else:
        config = {}

    model = Model(vocab_config, **config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_dl))
    trainner = Trainer(
        model, optimizer, train_dl, test_dl, 
        device=args.device, scheduler=sched,
        log_dir=args.log_dir,
        weight_dir=args.weight_dir
    )

    print("Start training")
    trainner.train(args.num_epoch)
