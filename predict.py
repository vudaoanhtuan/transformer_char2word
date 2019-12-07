import json
import argparse

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn

from model import Model
from dataset import Dataset, Tokenizer, load_tokenizer, MaskDataset
from decode import greedy_decode


parser = argparse.ArgumentParser()
parser.add_argument('--src_vocab', required=True)
parser.add_argument('--tgt_vocab', required=True)
parser.add_argument('--test_file', required=True)
parser.add_argument('--model_weight', required=True)
parser.add_argument('--model_config')
parser.add_argument('--device', default='cpu')

if __name__ == "__main__":
    args = parser.parse_args()

    print("Load vocab")
    tokenizer = load_tokenizer(args.src_vocab, args.tgt_vocab)

    print("Init model")
    src_vocab_len = len(tokenizer.src_stoi)
    tgt_vocab_len = len(tokenizer.tgt_stoi)

    if args.model_config:
        with open(args.model_config) as f:
            config = json.load(f)
    else:
        config = {}

    model = Model(src_vocab_len, tgt_vocab_len, **config)
    print("Load model")
    state = torch.load(args.model_weight)
    model.load_state_dict(state)

    df = pd.read_csv(args.test_file)
    predict = []
    for s in tqdm(df['source'].values):
        p = greedy_decode(model, tokenizer, s)
        predict.append(p)

    df['predict'] = predict
    df.to_csv(args.test_file+"predict", index=False)


