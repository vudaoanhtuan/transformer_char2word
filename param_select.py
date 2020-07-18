import os
import argparse
import json
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trainner import Trainer
from model import Model
from dataset import SingleDataset
from tokenizer import Tokenizer, load_tokenizer
from decode import greedy_decode, BeamDecode
from utils.wer import wer

parser = argparse.ArgumentParser()
parser.add_argument('--src_vocab', required=True)
parser.add_argument('--tgt_vocab', required=True)
parser.add_argument('--test_file', required=True)
parser.add_argument('--model_weight', required=True)
parser.add_argument('--model_config')
parser.add_argument('--device', default='cpu')
parser.add_argument('--lm_path')


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

    model = Model(src_vocab_len, tgt_vocab_len, init_param=False, **config)
    model.load_state_dict(torch.load(args.model_weight, map_location=torch.device('cpu')))
    model.eval()
    beam_decoder = BeamDecode(
        model, tokenizer, 
        beam_size=5, max_len=50, pc_min_len=0.8,
        lm_path=args.lm_path, alpha=0.45, 
        len_norm_alpha=1.2
    )
    with open(args.test_file) as f:
        data = f.read().split("\n")[:-1]
    
    alphas = []
    wers = []
    print("Start")
    for cur_alpha in tqdm(list(range(0, 101, 5))):
        alphas.append(cur_alpha*0.01)
        total_err = 0
        total_word = 0
        for line in data:
            line = line.split("|")
            s = line[0]
            l = line[1]
            pred = beam_decoder.predict_topk(
                s, 
                beam_size=3, 
                alpha=cur_alpha*0.01,
                pc_min_len=0.8, len_norm_alpha=1.2,
                post_process=False, re_scale=False
            )
            try:
                pred = pred[0][1]
            except:
                pred = ''
            num_err, sent_len = wer(pred, l)
            total_err += num_err
            total_word += sent_len
        err = total_err*1.0/total_word
        wers.append(err)
        print("%.2f | %.4f" % (cur_alpha*0.01, err))
    
    df = pd.DataFrame({
        'alpha': alphas,
        'wer': wers
    })
    
    df.to_csv("param_select.csv", index=False)


