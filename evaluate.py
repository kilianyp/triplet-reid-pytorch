import argparse
import subprocess
import sys
import os
from embed import run
import gc
from embed import make_dataset_default

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Which dataset to evaluate", required=True)
parser.add_argument("--model", help="Path to model", required=True)
parser.add_argument("--query", help="Path to query csv", required=True)
parser.add_argument("--gallery", help="Path to gallery csv", required=True)
parser.add_argument("--data_dir", help="Path to image directory", required=True)
parser.add_argument("--batch_size", help="Batch size", type=str, default="32")
parser.add_argument("--augmentation", required=True)
parser.add_argument("--prefix", required=False)
args = parser.parse_args()
model = args.model


# calculate embeddings

gallery_csv= os.path.expanduser(args.gallery)
query_csv = os.path.expanduser(args.query)
data_dir = os.path.expanduser(args.data_dir)

gallery_embeddings = run(gallery_csv, data_dir, model, 4, 
                         make_dataset_default, args.augmentation, args.prefix)
# generated filename is written in stderr, remove some whitecharacters.

if gallery_csv == query_csv:
    query_embeddings = gallery_embeddings
else:
    query_embeddings = run(query_csv, data_dir, model, 4, make_dataset_default, 
                           args.augmentation, args.prefix)
print("Evaluating query: {}, gallery {}".format(query_csv, gallery_csv))
eval_args = ["python3", "/home/pfeiffer/Projects/cupsizes/evaluate.py",
             "--dataset", args.dataset,
             "--query_dataset", query_csv,
             "--query_embeddings", query_embeddings,
             "--gallery_dataset", gallery_csv,
             "--gallery_embeddings", gallery_embeddings,
             "--metric", "euclidean",
             "--batch_size", args.batch_size]


file_name =  "{}{}_{}_{}.txt".format(args.prefix, os.path.basename(query_csv), os.path.basename(gallery_csv), os.path.basename(model))

txt_file = os.path.join(os.path.dirname(model), file_name)
print(txt_file)
# free pytorch memory
gc.collect()
with open(txt_file, 'w') as f_h:
    print(' '.join(eval_args))
    task = subprocess.Popen(eval_args, stdout=subprocess.PIPE, universal_newlines=True)
    for line in task.stdout:
        sys.stdout.write(line)
        f_h.write(line)
    task.wait()

while True:
    delete = input("Do you want to delete the embedding files? Y/n \n"
                   "({}|{})".format(query_embeddings, gallery_embeddings))
    if delete.lower() == "y":
        os.remove(query_embeddings)
        os.remove(gallery_embeddings)
        print("Deleted: {},{}".format(query_embeddings, gallery_embeddings))
        break
    elif delete.lower() == 'n':
            break
    else:
        print("Invalid input.")
