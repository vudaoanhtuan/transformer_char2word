import numpy as np
import torch
import kenlm

from model import generate_square_subsequent_mask

def greedy_decode(model, tokenizer, inp, max_len=100):
    model.eval()
    src = tokenizer.tokenize_src(inp)
    src = torch.tensor([src]).long()
    memory = model.encode(src)
    
    hyp = torch.zeros(1, max_len).long()
    hyp[0,0] = tokenizer.tgt_stoi['[bos]']
    
    for i in range(1, max_len):
        tgt_mask = generate_square_subsequent_mask(i) # TxT
        out = model.decode(hyp[:,:i], memory, tgt_mask=tgt_mask)
        out = out.argmax(dim=-1)
        w_ix = out[0,-1]
        if w_ix == tokenizer.tgt_stoi['[eos]']:
            break
        hyp[0,i] = w_ix
    
    seq = hyp.numpy().tolist()[0]
    sent = [tokenizer.tgt_itos[x] for x in seq[1:i]]
    return ' '.join(sent)


class BeamDecode():
    def __init__(self, model, tokenizer, beam_size=10, max_len=50, pc_min_len=0.8, lm_path=None, alpha=0.0, len_norm_alpha=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_len = max_len
        self.alpha = alpha
        if lm_path:
            self.lm = kenlm.Model(lm_path)
        else:
            self.lm = None
            self.alpha = 0.0
        self.len_norm_alpha = len_norm_alpha
        self.pc_min_len = pc_min_len
    
    def ix_to_sent(self, ixs):
        sent = [self.tokenizer.tgt_itos[x] for x in ixs]
        return ' '.join(sent)

    def beam_search(self, src):

        src_len = src.count(" ") + 1
        min_len = int(self.pc_min_len*src_len)

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
            n_hyps = min(self.beam_size, len(hyps))

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
                    lm_score = 0;
                    if self.lm:
                        lm_score = self.lm.score(sent, eos=False)
                    combined_score = self.alpha * lm_score + (1-self.alpha) * (log_scores[h] + kv)
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
                if i_word == self.tokenizer.eos and ix > min_len: # eos at second token
                    len_norm_score = ((5.0 + len(h)) / 6.0) ** self.len_norm_alpha
                    completed_sent.append((i_combined_score / len_norm_score, h))
                else:
                    new_hyps.append(h)
                    new_log_score.append(i_score)
                    num_hyp += 1
                i += 1

            hyps = new_hyps
            log_scores = new_log_score
            if len(completed_sent) > self.beam_size:
                break
        completed_sent.sort(reverse=True)
        return completed_sent[:self.beam_size]
    
    def predict(self, src):
        pred = self.beam_search(src)[0][1][1:-1]
        sent = self.ix_to_sent(pred)
        return sent

        # candidates = self.beam_search(src)
        # candidates_len = [len(x[1]) for x in candidates]
        # best_len = np.quantile(candidates_len, 0.8)
        # for i,l in enumerate(candidates_len):
        #     if l >= int(best_len):
        #         sent_seq = candidates[i][1][1:-1]
        #         sent = self.ix_to_sent(sent_seq)
        #         return sent
