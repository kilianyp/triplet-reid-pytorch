import argparse
import subprocess
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--use_junk", action='store_true')
parser.add_argument("--detector", choices=["DPM", "SDP", "FRCNN"], required=True)
args = parser.parse_args()
model = args.model
print(model)

eval_output_folder = "evaluations"

# calculate embeddings

data_dir="/work/pfeiffer/MOT17/ReID/{}".format(args.detector)
if args.use_junk:
    gallery_csv="/work/pfeiffer/MOT17/ReID/{}/gallery.csv".format(args.detector)
else:
    gallery_csv="/work/pfeiffer/MOT17/ReID/{}/val.csv".format(args.detector)
query_csv="/work/pfeiffer/MOT17/ReID/{}/val.csv".format(args.detector)
junk_csv="/work/pfeiffer/MOT17/ReID/{}/junk.csv".format(args.detector)



gallery_csv= os.path.expanduser(gallery_csv)
query_csv = os.path.expanduser(query_csv)


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
             "--dataset", "diagonal",
             "--query_dataset", query_csv,
             "--query_embeddings", query_embeddings,
             "--gallery_dataset", gallery_csv,
             "--gallery_embeddings", gallery_embeddings,
             "--metric", "euclidean",
             "--batch_size", "128"]
eval_output_folder = os.path.join(
    eval_output_folder,
    "{}_{}".format(os.path.basename(query_csv), os.path.basename(gallery_csv)))
print(eval_output_folder)

if not os.path.isdir(eval_output_folder):
    os.mkdir(eval_output_folder)
txt_file = os.path.join(eval_output_folder, "{}.txt".format(os.path.basename(model)))
print(txt_file)
with open(txt_file, 'w') as f_h:
    task = subprocess.Popen(eval_args, stdout=f_h)
    task.wait()
