import math
from pathlib import Path

import os
import random
import h5py

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from sklearn.model_selection import StratifiedKFold


class Hdf5Dataset(data.Dataset):
    def __init__(
        self,
        path="/home/ubuntu/data/asahi_changedet/data/hdf5_dataset_v1.h5",
        # path="/home/ubuntu/local/hdf5_dataset_v1.h5",
        transform=None,
        fold_index=0,
        mode="train",
        img_type=("aero"),
        split_dataset=False,
    ):
        hdf_path = Path(path)
        hdf = h5py.File(hdf_path, "r")
        dataset_info = hdf["dataset_info"]
        if split_dataset:
            self._data_list = np.array(dataset_info["training_th_list"])
            label_list = np.array(list(dataset_info["label_th_list"]))
        else:
            self._data_list = np.array(dataset_info["training_list"])
            label_list = np.array(list(dataset_info["label_list"]))
        print("total")
        self.count_label(label_list)
        self._n_split = int(dataset_info["n_split"][0])
        # self._data_list = np.array(list(hdf["training_list"]["training_list"]))
        self._data = hdf["train"]
        df_fold = set_fold(label_list)
        if mode == "train":
            print("train")
            mask = df_fold.fold != fold_index
            self._data_list = self._data_list[mask]
            self.count_label(label_list[mask])
        else:
            print("test")
            mask = df_fold.fold == fold_index
            self._data_list = self._data_list[mask]
            # self._data_list = self._data_list[~mask]
            self.count_label(label_list[mask])
        self._transform = transform
        self._img_type = img_type

    def __getitem__(self, idx):
        name = self._data_list[idx]
        x = int(name[7:-4].decode()) % self._n_split
        data = self._data[str(x)][name]
        input_img = self.get_img(data)
        label = data["label"][0]
        label = torch.tensor(label).float()
        # label = torch.tensor(label).long()
        ret = {"data": input_img, "label": label, "name": name.decode()}
        # ret = {"diff_img": diff_img, "label": label}
        return ret
        # return diff_img, label

    def __len__(self):
        return len(self._data_list)

    def get_img(self, data):
        diff_img_list = []
        diff_img_dict = {}
        src_img_dict = {}
        for t in self._img_type:
            diff_img, [new_img, old_img] = self.get_diff_img(data, t)
            diff_img_dict[t] = diff_img
            src_img_dict[t + "_new"] = new_img
            src_img_dict[t + "_old"] = old_img
            # diff_img_list.append(diff_img)
        # diff_img = np.concatenate(diff_img_list)
        input_img = {"diff_img": diff_img_dict, "src_img": src_img_dict}
        return input_img

    def get_diff_img(self, data, img_type):
        if img_type == "aero":
            new_aero_img = Image.fromarray(np.array(data["new_aero"]))
            old_aero_img = Image.fromarray(np.array(data["old_aero"]))
        elif img_type == "ref":
            new_aero_img = Image.fromarray(np.array(data["new_ref"])).convert("L")
            old_aero_img = Image.fromarray(np.array(data["old_ref"])).convert("L")
        elif img_type == "ele":
            new_aero_img = Image.fromarray(np.array(data["new_ele"])).convert("L")
            old_aero_img = Image.fromarray(np.array(data["old_ele"])).convert("L")
        elif img_type == "shade":
            new_aero_img = Image.fromarray(np.array(data["new_shade"])).convert("L")
            old_aero_img = Image.fromarray(np.array(data["old_shade"])).convert("L")
        new_aero_img = self._transform(new_aero_img)
        old_aero_img = self._transform(old_aero_img)
        diff_img = new_aero_img - old_aero_img
        diff_img = diff_img**2
        diff_img = torch.sqrt(diff_img)
        return diff_img, [new_aero_img, old_aero_img]

    def count_label(self, label):
        label = np.array(label)
        num_neg = label == 0
        num_pos = label == 1
        print(f"num sample  : [{label.shape[0]}]")
        print(f"num positive: [{sum(num_pos)}]")
        print(f"num negative: [{sum(num_neg)}]")
        print("---------------------------")


class AsahiDataset(data.Dataset):
    def __init__(
        self,
        mode="train",
        csv_dir="./data/result_mini.csv",
        transform=None,
        fold_index=0,
        aero_dir="data/aerophoto",
        elevation_dir="data/elevation",
        refrection_dir="data/refrection",
        shade_dir="data/shaded-relief",
    ):
        df = pd.read_csv(csv_dir, index_col=0).reset_index()
        df_fold = set_fold(df.label.values)
        if mode == "train":
            ind = df_fold.fold != fold_index
            df = df[ind]
        else:
            ind = df_fold.fold == fold_index
            df = df[ind]
        self._df = df
        self._transform = transform
        self._name = self._df.name.values
        self._label = self._df.label.values
        self._aero_dir = aero_dir
        self._ele_dir = elevation_dir
        self._ref_dir = refrection_dir
        self._shade_dir = shade_dir

    def __getitem__(self, idx):
        name = self._name[idx]
        label = self._label[idx]
        diff_img, new, old = self.get_img(self._aero_dir, name)
        diff_img = torch.cat([diff_img, new, old], 0)
        if label == "blue":
            label = 0
        else:
            label = 1
        label = torch.tensor(label).long()
        exit()
        return diff_img, label

    def __len__(self):
        return len(self._df)

    def get_img(self, path, name):
        new_name = os.path.join(path, "ortho_new/192", name)
        new_aero_img = Image.open(new_name)
        new_aero_img = self._transform(new_aero_img)
        old_name = os.path.join(path, "ortho_old/192", name)
        old_aero_img = Image.open(old_name)
        old_aero_img = self._transform(old_aero_img)
        diff_img = new_aero_img - old_aero_img
        return diff_img, new_aero_img, old_aero_img


def set_fold(labels, n_split=5):
    skf = StratifiedKFold(n_splits=n_split)
    df_folds = pd.DataFrame({"labels": labels})
    df_folds.loc[:, "fold"] = 0
    for fold_number, (train_index, val_index) in enumerate(
        skf.split(X=df_folds.index, y=df_folds.labels)
    ):
        df_folds.loc[df_folds.iloc[val_index].index, "fold"] = fold_number
    return df_folds


if __name__ == "__main__":
    from transform import return_img_transform

    h = Hdf5Dataset()
    h[0]
    exit()

    t = return_img_transform()
    d = AsahiDataset(transform=t)
    for i in d:
        print(i)
        exit()
