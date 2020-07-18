import argparse
import numpy as np
from utils.word_transform import transform_sentence

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', required=True)
parser.add_argument('--out_file', required=True)
parser.add_argument('--num_sent', type=int, default=2000)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.test_file) as f:
        sents = f.read().split("\n")[:-1]
    ids = np.random.choice(list(range(len(sents))), args.num_sent, False)
    with open(args.out_file, 'w') as f:
        for ix in ids:
            sent = sents[ix]
            sent = sent.strip()
            wrong_sent = transform_sentence(sent)
            f.write("%s|%s\n"%(wrong_sent, sent))
