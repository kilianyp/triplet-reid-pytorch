from torch.utils.data import Dataset
import sys
sys.path.append("../mot17")
from mot_dataset_creator import MotGtDataReader
from csv_dataset import mot_name_fn
from csv_dataset import pil_loader_with_crop
import numpy as np
class MotGtDataset(Dataset):
    def __init__(self, mot_dir, visibility, transform=None, limit=None, rewrite=True):
        # sample per sequence?
        # should be done by the sampler
        # but supported here
        reader = MotGtDataReader()
        self.data = reader.read_in_data(mot_dir)

        # clean data
        self.data.filter(self.data.data["class"].isin([1, 2, 7]),
                         "classes")



        self.data.data["pid"] = self.data.data["seq_id"].map(str) + '-' + self.data.data["pid"].map(str)

        labels = self.data.data["pid"].unique()
        if rewrite:
            label_dic = {}
            new_label = 0
            # rewrite pids starting from 0
            for label in labels:
                label_dic[label] = new_label
                new_label += 1
           
            new_pids = []
            for row in self.data.data.itertuples():
                label = label_dic[row.pid]
                new_pids.append(label)
            
            self.data.data["pid"] = np.asarray(new_pids)

        self.num_labels = len(labels)

        self.data.filter(self.data.data["visibility"] < 0.2)

        self.loader = pil_loader_with_crop

    def __getitem__(self, index):
        """Index is tuple of dataset, index."""

        row = self.iloc[index]
        img = self.loader(mot_name_fn(row))

        target = row["pid"]
        if self.transform is not None:
            img = self.transform(img, row)

        return img, target, row

    def __len__(self):
        return sum(map(len, self.datasets))
