from model import generate_square_subsequent_mask

def greedy_decode(model, tokenizer, inp, max_len=50):
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
    sent = [tokenizer.tgt_itos[x] for x in seq[1:]]
    return ' '.join(sent)