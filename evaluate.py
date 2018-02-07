import argparse
import subprocess
import sys
import os
from embed import create_embeddings

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Which dataset to evaluate", required=True)
parser.add_argument("--model", help="Path to model", required=True)
parser.add_argument("--query", help="Path to query csv", required=True)
parser.add_argument("--gallery", help="Path to gallery csv", required=True)
parser.add_argument("--data_dir", help="Path to image directory", required=True)
parser.add_argument("--prefix", required=False)
args = parser.parse_args()
model = args.model


# calculate embeddings

gallery_csv= os.path.expanduser(args.gallery)
query_csv = os.path.expanduser(args.query)
data_dir = os.path.expanduser(args.data_dir)


gallery_embeddings = create_embeddings(gallery_csv, data_dir, model)
# generated filename is written in stderr, remove some whitecharacters.

if gallery_csv == query_csv:
    query_embeddings = gallery_embeddings
else:
    query_embeddings = create_embeddings(query_csv, data_dir, model)

print("Evaluating query: {}, gallery {}".format(query_csv, gallery_csv))
eval_args = ["python3", "/home/pfeiffer/Projects/cupsizes/evaluate.py",
             "--dataset", args.dataset,
             "--query_dataset", query_csv,
             "--query_embeddings", query_embeddings,
             "--gallery_dataset", gallery_csv,
             "--gallery_embeddings", gallery_embeddings,
             "--metric", "euclidean",
             "--batch_size", "32"]


file_name =  "{}{}_{}_{}.txt".format(args.prefix, os.path.basename(query_csv), os.path.basename(gallery_csv), os.path.basename(model))

txt_file = os.path.join(os.path.dirname(model), file_name)
print(txt_file)
with open(txt_file, 'w') as f_h:
    print(' '.join(eval_args))
    task = subprocess.Popen(eval_args, stdout=f_h)
    task.wait()
