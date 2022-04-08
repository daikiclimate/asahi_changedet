import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from addict import Dict
from barbar import Bar

from dataset import return_dataset
from evaluator import evaluator
from models import build_model


def get_arg():
    parser = argparse.ArgumentParser(description="image model for action segmentation")
    parser.add_argument("config", type=str, help="path of a config file")
    args = parser.parse_args()
    config = Dict(yaml.safe_load(open(args.config)))
    return config


def main():
    config = get_arg()
    config.save_folder = os.path.join(config.save_folder, config.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_set = return_dataset.return_dataloader(config)
    model = build_model.build_model(config)
    model = model.to(device)
    path = os.path.join(config.save_folder, "model.pth")
    model.load_state_dict(torch.load(path))
    model.eval()
    labels = []
    preds = []
    names = []
    prob = []
    eval = evaluator()
    th = 0.1
    for data in Bar(test_set):
        output = model(data["data"], device)["out"]
        output = torch.sigmoid(output)
        prob.extend(output.detach().cpu().numpy())
        output[output < th] = 0
        output[output >= th] = 1
        label = data["label"]

        labels.extend(label.detach().cpu().numpy())
        preds.extend(output.detach().cpu().numpy())
        names.extend(data["name"])

    labels, preds = np.array(labels), np.array(preds)
    judge = np.zeros(len(labels))
    judge[labels == preds] = 1
    result_df = pd.DataFrame()
    result_df["name"] = names
    result_df["label"] = labels
    result_df["pred"] = preds
    result_df["judge"] = judge
    result_df["prob"] = prob
    result_df.to_csv("result/output.csv")
    # with open("test.pkl", "wb") as f:
    # pickle.dump(preds, f)
    eval.set_data(labels, preds)
    eval.print_eval(["accuracy"])
    score = eval.return_eval_score()


if __name__ == "__main__":
    main()
