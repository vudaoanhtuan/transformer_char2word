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

    df = pd.read_csv(args.bisen_file, sep='\t', names=['src', 'tgt'])
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.random_state)

    train_df.to_csv(args.train_file, index=False, header=False, sep='\t')
    test_df.to_csv(args.test_file, index=False, header=False, sep='\t')
