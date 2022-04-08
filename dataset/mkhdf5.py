import glob
import cv2
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tqdm
from PIL import Image


def diff_img(a, b):
    a = a.reshape(-1).astype(np.float)
    b = b.reshape(-1).astype(np.float)
    diff = a - b
    sq_diff = diff**2
    sq_diff_sum = np.mean(sq_diff)
    return np.sqrt(sq_diff_sum)


if __name__ == "__main__":
    h5_path = "/home/ubuntu/data/asahi_changedet/data/hdf5_dataset_v2.h5"

    aero_old_path = "../data/aerophoto/ortho_old/192"
    aero_new_path = "../data/aerophoto/ortho_new/192"
    # aero_old = os.listdir(aero_old_path)

    refrection_old_path = "../data/refrection/ortho_old/192"  # rtho_old/192"
    refrection_new_path = "../data/refrection/ortho_new/192"

    elevation_old_path = "../data/elevation/ortho_old/192"  # rtho_old/192"
    elevation_new_path = "../data/elevation/ortho_new/192"

    shaded_relief_old_path = "../data/shaded-relief/ortho_old/192"  # rtho_old/192"
    shaded_relief_new_path = "../data/shaded-relief/ortho_new/192"
    df = pd.read_csv("../data/shape/R020101-R030101/R020101-R030101.csv")
    df_list = df.TEXT.values
    n_sample = len(df)
    df_list = df_list[:n_sample]
    df_label = df["判定"].values
    df_label = df_label[:n_sample]
    with h5py.File(h5_path, "w") as h5:
        train_group = h5.create_group("train")
        n_split = 1000
        groups = [train_group.create_group(str(i)) for i in range(n_split)]
        training_list = []
        training_th_list = []
        lost_name = []
        label_list = []
        label_th_list = []
        aero_th = 30
        ref_th = 40
        for idx, (name, label) in tqdm.tqdm(enumerate(zip(df_list, df_label))):
            value = int(name[7:]) % n_split
            name = name + ".png"
            old_tmp_1 = cv2.imread(os.path.join(aero_old_path, name), 1)
            new_tmp_1 = cv2.imread(os.path.join(aero_new_path, name), 1)
            if not isinstance(old_tmp_1, np.ndarray):
                lost_name.append(name)
                print(lost_name)
                exit()
                continue
            else:
                training_list.append(name)
                label_list.append(label)
            d_aero = diff_img(new_tmp_1, old_tmp_1)

            subset_group = groups[value].create_group(name)
            # subset_group = train_group.create_group(name)
            subset_group.create_dataset("label", data=[label])
            subset_group.create_dataset("new_aero", data=old_tmp_1)
            subset_group.create_dataset("old_aero", data=new_tmp_1)
            subset_group.create_dataset("diff_aero", data=d_aero)

            old_tmp_1 = cv2.imread(os.path.join(refrection_old_path, name), 1)
            new_tmp_1 = cv2.imread(os.path.join(refrection_new_path, name), 1)
            d_ref = diff_img(new_tmp_1, old_tmp_1)
            subset_group.create_dataset("new_ref", data=old_tmp_1)
            subset_group.create_dataset("old_ref", data=new_tmp_1)
            subset_group.create_dataset("diff_ref", data=d_ref)

            old_tmp_1 = cv2.imread(os.path.join(elevation_old_path, name), 1)
            new_tmp_1 = cv2.imread(os.path.join(elevation_new_path, name), 1)
            d_ele = diff_img(new_tmp_1, old_tmp_1)
            subset_group.create_dataset("new_ele", data=old_tmp_1)
            subset_group.create_dataset("old_ele", data=new_tmp_1)
            subset_group.create_dataset("diff_ele", data=d_ele)

            old_tmp_1 = cv2.imread(os.path.join(shaded_relief_old_path, name), 1)
            new_tmp_1 = cv2.imread(os.path.join(shaded_relief_new_path, name), 1)
            d_shade = diff_img(new_tmp_1, old_tmp_1)
            subset_group.create_dataset("new_shade", data=old_tmp_1)
            subset_group.create_dataset("old_shade", data=new_tmp_1)
            subset_group.create_dataset("diff_shade", data=d_shade)
            # if label == 1:
            #    print(f"{label}, [{d_aero}, {aero_th}], [{d_ref}, {ref_th}]")
            if (d_aero > aero_th) & (d_ref > ref_th):
                training_th_list.append(name)
                label_th_list.append(label)
            # if idx == 500:
            #    break
        dataset_info = h5.create_group("dataset_info")
        dataset_info.create_dataset("training_list", data=training_list)
        # training_group = h5.create_group("training_th_list")
        dataset_info.create_dataset("training_th_list", data=training_th_list)
        dataset_info.create_dataset("label_list", data=label_list)
        dataset_info.create_dataset("label_th_list", data=label_th_list)
        dataset_info.create_dataset("n_split", data=[n_split])
        print("done")
