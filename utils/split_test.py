import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("bisen_file")
parser.add_argument("--train_file", default="train.tsv")
parser.add_argument("--test_file", default="test.tsv")
parser.add_argument("--test_size", default=0.2, type=float)
parser.add_argument("--random_state", default=42, type=int)

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.bisen_file) as f:
        corpus = f.readlines()
    
    train, test = train_test_split(corpus, test_size=args.test_size, random_state=args.random_state)

    with open(args.train_file, 'w') as f:
        f.writelines(train)

    with open(args.test_file, 'w') as f:
        f.writelines(test)
