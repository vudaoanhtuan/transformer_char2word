# transformer_char2word

## Training
use `python3 train.py` with below required arguments:  
- src_vocab: source vocabulary, each word per line  
- tgt_vocab: target vocabulary, each word per line  
- train_file: TSV file with 2 columns, first column contains source sentences, second column contains target sentences  
- test_file: same as train_file  
- model_config: configuration of model, view `model_config.json`  

## Predict
view `predict.py` for more infomation

## Note
- Currently, the tokenizer will tokenize source sentences at character level, modify `tokenizer.py` to tokenize sentences at word level  
- Do not use MaskDataset in `dataset.py`, it is not correct. Use Dataset instead  
