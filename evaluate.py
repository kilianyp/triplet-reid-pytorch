import argparse
import subprocess
import sys
import os

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


gallery_args = ["python3","embed.py",
                "--csv_file", gallery_csv,
                "--data_dir", data_dir,
                "--model", model]
task = subprocess.Popen(gallery_args, stderr=subprocess.PIPE)
# generated filename is written in stderr, remove some whitecharacters.
gallery_embeddings = task.stderr.read().rstrip()
gallery_embeddings = gallery_embeddings.decode('utf-8')
#gallery_embeddings = "/home/pfeiffer/Projects/triplet-reid-pytorch/embed/train_BatchHard-soft_18-4_0.000300_25000/val_model_25000_embeddings.h5"
task.wait()
if(task.returncode > 0):
    print(gallery_embeddings)
    print("Error %d!" % task.returncode)
    sys.exit()

if gallery_csv == query_csv:
    query_embeddings = gallery_embeddings
else:
    query_args = ["python3", "embed.py",
                  "--csv_file", query_csv,
                  "--data_dir", data_dir,
                  "--model", model]
    task = subprocess.Popen(query_args, stderr=subprocess.PIPE)
    query_embeddings = task.stderr.read().rstrip().decode("utf-8")
    print(query_embeddings)
    task.wait()
    if(task.returncode > 0):
        print("Error %d!" % task.returncode)
        sys.exit()

print("Evaluating query: {}, gallery {}".format(query_csv, gallery_csv))
eval_args = ["python3", "/home/pfeiffer/Projects/cupsizes/evaluate.py",
             "--dataset", args.dataset,
             "--query_dataset", query_csv,
             "--query_embeddings", query_embeddings,
             "--gallery_dataset", gallery_csv,
             "--gallery_embeddings", gallery_embeddings,
             "--metric", "euclidean",
             "--batch_size", "128"]


file_name =  "{}{}_{}_{}.txt".format(args.prefix, os.path.basename(query_csv), os.path.basename(gallery_csv), os.path.basename(model))

txt_file = os.path.join(os.path.dirname(model), file_name)
print(txt_file)
with open(txt_file, 'w') as f_h:
    print(' '.join(eval_args))
    task = subprocess.Popen(eval_args, stdout=f_h)
    task.wait()
