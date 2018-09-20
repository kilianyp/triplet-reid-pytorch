# Description
A pytorch implementation of the "In Defense of the Triplet Loss for Person Re-Identification" paper (https://arxiv.org/abs/1703.07737).

This repository also contains also a implementation of the MGN network from the paper: 
["Learning Discriminative Features with Multiple Granularities for Person Re-Identification"](https://arxiv.org/abs/1804.01438)

It is built from scratch with ease in experimenting, modification and reproducability.
However, it still work in progress.

# Requirements

- numpy
- Pillow
- h5py
- scipy
- torch
- torchvision

This repository depends for evaluation on the original repository:
https://github.com/VisualComputingInstitute/triplet-reid

For evaluation:
- tensorflow
- scikit-learn
- scipy
- h5py

# Installation
git clone https://github.com/kilsenp/triplet-reid-pytorch.git
pip install -r requirements.txt (for training)

# Train
```
python3 train.py --csv_file path/to/ 
                 --data_dir path/to/image/base/directory 
                 --loss BatchHard
                 --model trinet
```
The script looks for the image files by joining the data_dir and image path given in the csv file.
The csv file should be a two column file with pids, fids.

For market, you can find them [here](https://github.com/VisualComputingInstitute/triplet-reid/tree/master/data):


If you would like to train using the MGN network, use the following command:
```
python3 train.py --csv_file path/to/ 
                 --data_dir path/to/image/base/directory 
                 --loss BatchHardWithSoftmax
                 --model mgn
                 --mgn branches 1 2
                 --dim 256
```


# Evaluation

You can use embed.py to write out embeddings that are compatible with the 
evaluation script.

```
python3 embed.py --csv_file path/to/gallery/or/query/csv
                 --data_dir path/to/image/base/directory
                 --model path/to/model/file
```                 
To calculate the final scores, please use the evaluation script from 
[here](https://github.com/VisualComputingInstitute/triplet-reid#evaluating-embeddings)!

# Scores (and pretrained models)              
### Market-1501

Settings: 
- P=18 
- K=4
- dim=128

Download Model ([GoogleDrive](https://drive.google.com/open?id=1eNJuLxRz3dJ0MkVjoLP6vshxZUn_NLn0))

|Experiment| mAP | top-1 | top-5| top-10|
|---|---:|---:|---:|---:|
| Without TenCrop| 65.06% | 80.31% | 92.25% | 94.71% |
| With TenCrop |  69.44% | 83.40% | 93.59% | 96.17% |


# TODO
- [] Evaluate current MGN
- [] Improve logging (Use tensorboard or similar)
- [] Clean up CPU GPU mess






    
