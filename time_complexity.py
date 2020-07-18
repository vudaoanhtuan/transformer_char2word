import os
import argparse
import json
import time

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
    
    sentence = "mot buoi chieu dep troi toi di dao trong cong \
        vien gan truong thi bat gap mot cu gia dang \
        ngoi tren ghe da dang lam mot viec gi do".split()
    
    # No LM
    sent_len = []
    beam_size = []
    run_time = []
    for bs in tqdm(list(range(1,11))):
        for sl in tqdm(list(range(2,31,2))):
            com_time = 0
            sent = ' '.join(sentence[:sl])
            for i in range(5):
                start_time = time.time()
                pred = beam_decoder.predict_topk(
                    sent, 
                    beam_size=bs, 
                    alpha=0,
                    pc_min_len=0.8, len_norm_alpha=1.2
                )
                end_time = time.time()
                rt = end_time - start_time
                com_time += rt
            com_time = com_time / 5
            sent_len.append(sl)
            beam_size.append(bs)
            run_time.append(com_time)
    
    df_no_lm = pd.DataFrame{
        'beam_size': beam_size,
        'sent_len': sent_len,
        'run_time': run_time
    }

    df_no_lm.to_csv('time.nolm.csv', index=False)

    # LM
    sent_len = []
    beam_size = []
    run_time = []
    for bs in tqdm(list(range(1,11))):
        for sl in tqdm(list(range(2,31,2))):
            com_time = 0
            sent = ' '.join(sentence[:sl])
            for i in range(5):
                start_time = time.time()
                pred = beam_decoder.predict_topk(
                    sent, 
                    beam_size=bs, 
                    alpha=0.3,
                    pc_min_len=0.8, len_norm_alpha=1.2
                )
                end_time = time.time()
                rt = end_time - start_time
                com_time += rt
            com_time = com_time / 5
            sent_len.append(sl)
            beam_size.append(bs)
            run_time.append(com_time)
    
    df_lm = pd.DataFrame{
        'beam_size': beam_size,
        'sent_len': sent_len,
        'run_time': run_time
    }

    df_lm.to_csv('time.lm.csv', index=False)