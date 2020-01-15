import json
import argparse

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn

from model import Model
from dataset import Dataset, MaskDataset
from tokenizer import Tokenizer, load_tokenizer
from decode import greedy_decode, BeamDecode


parser = argparse.ArgumentParser()
parser.add_argument('--src_vocab', required=True)
parser.add_argument('--tgt_vocab', required=True)
parser.add_argument('--test_file', required=True)
parser.add_argument('--model_weight', required=True)
parser.add_argument('--model_config')
parser.add_argument('--beam_size', type=int, default=10)
parser.add_argument('--max_len', type=int, default=50)
parser.add_argument('--pc_min_len', type=float, default=0.8)
parser.add_argument('--lm_path')
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--len_norm_alpha', type=float, default=0)
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

    print("Init decoder")
    beam_decoder = BeamDecode(
        model, tokenizer, 
        beam_size=args.beam_size, max_len=args.max_len, pc_min_len=args.pc_min_len,
        lm_path=args.lm_path, alpha=args.alpha, 
        len_norm_alpha=args.len_norm_alpha
    )

    df = pd.read_csv(args.test_file)
    df = df.rename(columns={"predict": "source"})
    predict = []
    for s in tqdm(df['source'].values):
        # p = greedy_decode(model, tokenizer, s)
        p = beam_decoder.predict(s)
        predict.append(p)

    df['predict'] = predict
    df.to_csv(args.test_file+".predict", index=False)


