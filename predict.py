import json
import argparse

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn

from model import Model
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


def cal_wer(ref, hyp, ignore_case=True):
    if ignore_case:
        ref = ref.lower()
        hyp = hyp.lower()
    r = ref.split()
    h = hyp.split()
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    DEL_PENALTY=1 # Tact
    INS_PENALTY=1 # Tact
    SUB_PENALTY=1 # Tact
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0

    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
    wer_result = (numSub + numDel + numIns) * 1.0 / (len(r))
    return {'wer':wer_result, 'cor':numCor, 'sub':numSub, 'ins':numIns, 'del':numDel, 'len': len(r)}

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
    print("Load model")
    model.load_state_dict(torch.load(args.model_weight, map_location=torch.device('cpu')))
    model.eval()

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
        p = beam_decoder.predict(
            s,
            beam_size=args.beam_size, 
            max_len=args.max_len, 
            pc_min_len=args.pc_min_len,
            alpha=args.alpha, 
            len_norm_alpha=args.len_norm_alpha
        )
        predict.append(p)

    df['predict'] = predict

    err = {
        "wer": [],
        "cor": [],
        "sub": [],
        "ins": [],
        "del": [],
        "len": []
    }

    for pre, lbl in zip(df.predict.values, df.label.values):
        e = cal_wer(lbl, pre)
        err['wer'].append(e['wer'])
        err['cor'].append(e['cor'])
        err['sub'].append(e['sub'])
        err['ins'].append(e['ins'])
        err['del'].append(e['del'])
        err['len'].append(e['len'])

    print("lm_path:", args.lm_path)
    print("beam_size:", args.beam_size)
    print("max_len:", args.max_len)
    print("pc_min_len:", args.pc_min_len)
    print("alpha:", args.alpha)
    print("len_norm_alpha:", args.len_norm_alpha)
    print("Summary:")
    print("sub:", sum(err['sub']))
    print("ins:", sum(err['ins']))
    print("del:", sum(err['del']))
    wer_score = (sum(err['sub'])+sum(err['ins'])+sum(err['del'])) * 100.0 / (sum(err['len']))
    print("wer: ", wer_score)
    df.to_csv(args.test_file+".predict", index=False)


