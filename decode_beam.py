import math

import torch
import kenlm

from model import generate_square_subsequent_mask

def beam_search(src, model, tokenizer, beam_size=10, max_len=50, device=torch.device('cpu')):
    model.eval()
    src = tokenizer.tokenize_src(src)
    src = torch.tensor([src]).long()
    memory = model.encode(src)
    
    hyps = [[tokenizer.bos]]
    log_scores = [0]
    completed_sent = []

    for ix in range(1, max_len):
        tgt_mask = generate_square_subsequent_mask(ix) # TxT
        tgt_mask.to(device)
        ix_candidates = []
        n_hyps = beam_size
        if ix==1:
            n_hyps = 1
        for h in range(n_hyps):
            tgt_inp = torch.tensor([hyps[h]], dtype=torch.long).to(device)

            h_logit = model.decode(tgt_inp, memory, tgt_mask=tgt_mask).detach().cpu()
            h_prob = torch.softmax(h_logit, dim=-1)
            h_prob = h_prob[0,-1,:]
            h_log_prob = h_prob.log()
            k_val, k_idx = h_log_prob.topk(beam_size)
            k_val = k_val.numpy().tolist()
            k_idx = k_idx.numpy().tolist()
            for kv, ki in zip(k_val, k_idx):
                # (score, sent_index, word_index)
                #TODO: add language model score here
                ix_candidates.append((log_scores[h] + kv, h, ki))

        ix_candidates.sort(reverse=True)

        new_hyps = []
        new_log_score = []
        num_hyp = 0
        i = 0
        while num_hyp < beam_size and i < len(ix_candidates):
            i_score, i_sent, i_word = ix_candidates[i]
            h = hyps[i_sent].copy()
            h.append(i_word)
            if i_word == tokenizer.eos:
                completed_sent.append((i_score, h))
            else:
                new_hyps.append(h)
                new_log_score.append(i_score)
                num_hyp += 1
            i += 1

        hyps = new_hyps
        log_scores = new_log_score
    completed_sent.sort(reverse=True)
    return completed_sent[:beam_size]


class BeamDecode():
    def __init__(self, model, tokenizer, beam_size=10, max_len=50, lm_path=None, alpha=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_len = max_len
        self.lm = kenlm.Model(lm_path)
        self.alpha = alpha
    
    def ix_to_sent(self, ixs):
        sent = [self.tokenizer.tgt_itos[x] for x in ixs]
        return ' '.join(sent)

    def beam_search(self, src):
        self.model.eval()
        src = self.tokenizer.tokenize_src(src)
        src = torch.tensor([src]).long()
        memory = self.model.encode(src)
        
        hyps = [[self.tokenizer.bos]]
        log_scores = [0]
        completed_sent = []

        for ix in range(1, self.max_len):
            tgt_mask = generate_square_subsequent_mask(ix) # TxT
            tgt_mask
            ix_candidates = []
            n_hyps = self.beam_size
            if ix==1:
                n_hyps = 1
            for h in range(n_hyps):
                tgt_inp = torch.tensor([hyps[h]], dtype=torch.long)

                h_logit = self.model.decode(tgt_inp, memory, tgt_mask=tgt_mask).detach().cpu()
                h_prob = torch.softmax(h_logit, dim=-1)
                h_prob = h_prob[0,-1,:]
                h_log_prob = h_prob.log10()
                k_val, k_idx = h_log_prob.topk(self.beam_size)
                k_val = k_val.numpy().tolist()
                k_idx = k_idx.numpy().tolist()

                h_sent = self.ix_to_sent(hyps[h][1:])
                for kv, ki in zip(k_val, k_idx):
                    # (score, sent_index, word_index)
                    #TODO: add language self.model score here
                    if ki == self.tokenizer.eos:
                        sent = h_sent + " </s>"
                    else:
                        sent = h_sent + " " + self.tokenizer.tgt_itos[ki]
                    combined_score = self.alpha * self.lm.score(sent, eos=False) + (1-self.alpha) * (log_scores[h] + kv)
                    ix_candidates.append((combined_score, log_scores[h] + kv, h, ki))

            ix_candidates.sort(reverse=True)

            new_hyps = []
            new_log_score = []
            num_hyp = 0
            i = 0
            while num_hyp < self.beam_size and i < len(ix_candidates):
                i_combined_score, i_score, i_sent, i_word = ix_candidates[i]
                h = hyps[i_sent].copy()
                h.append(i_word)
                if i_word == self.tokenizer.eos:
                    completed_sent.append((i_combined_score, h))
                else:
                    new_hyps.append(h)
                    new_log_score.append(i_score)
                    num_hyp += 1
                i += 1

            hyps = new_hyps
            log_scores = new_log_score
        completed_sent.sort(reverse=True)
        return completed_sent[:self.beam_size]
    
    def predict(self, src):
        pred = self.beam_search(src)[0][1][1:-1]
        sent = self.ix_to_sent(pred)
        return sent





def greedy_decode(src, model, tokenizer, max_len=50):
    model.eval()
    src = tokenizer.tokenize_src(src)
    src = torch.tensor([src]).long()
    memory = model.encode(src)
    
    hyp = torch.zeros(1, max_len).long()
    hyp[0,0] = tokenizer.bos
    log_score = 0

    for i in range(1, max_len):
        tgt_mask = generate_square_subsequent_mask(i) # TxT
        logit = model.decode(hyp[:,:i], memory, tgt_mask=tgt_mask).detach().cpu()
        prob = torch.softmax(logit, dim=-1)
        prob = prob[0,-1,:]
        log_prob = prob.log()
        max_ix = log_prob.argmax().item()
        log_score += log_prob[max_ix].item()
        hyp[0,i] = max_ix
        if max_ix == tokenizer.eos:
            break
    
    seq = hyp.numpy().tolist()[0]
    sent = [tokenizer.tgt_itos[x] for x in seq[1:i]]
    return log_score, ' '.join(sent)


if __name__ == "__main__":
    import json
    import argparse

    import pandas as pd
    from tqdm import tqdm
    import torch
    import torch.nn as nn

    from model import Model
    from dataset import Dataset, MaskDataset
    from tokenizer import Tokenizer, load_tokenizer


    tokenizer = load_tokenizer('vocab/src_vocab.txt', 'vocab/tgt_vocab.txt')
    src_vocab_len = len(tokenizer.src_stoi)
    tgt_vocab_len = len(tokenizer.tgt_stoi)

    use_real_model = 1

    if use_real_model:
        print("Use real model")
        model = Model(src_vocab_len, tgt_vocab_len)
        state = torch.load("eval/model_v1.1/model.17.h5", map_location='cpu')
        model.load_state_dict(state)
    else:
        with open("data/temp/model.json") as f:
            config = json.load(f)
        model = Model(src_vocab_len, tgt_vocab_len, **config)

    src = 'xin chào the giới'

    beam_decode = BeamDecode(model, tokenizer, lm_path='lm/mini.bin', alpha=0.5)

    pred = beam_decode.predict(src)
    print(pred)
