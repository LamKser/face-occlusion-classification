from argparse import ArgumentParser

import pandas as pd
from sklearn.metrics import (accuracy_score, 
                             f1_score, 
                             precision_score, 
                             recall_score)

parser = ArgumentParser()
parser.add_argument("--f", type=str, required=True,
                    help="Results csv file")
args = parser.parse_args()

df = pd.read_csv(args.f)
print("Accuracy:", accuracy_score(df["labels"], df["preds"]))
print("Precision:", precision_score(df["labels"], df["preds"]))
print("Recall:", recall_score(df["labels"], df["preds"]))
print("F1:", f1_score(df["labels"], df["preds"]))