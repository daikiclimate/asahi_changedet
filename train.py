import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
import yaml
from addict import Dict
from barbar import Bar

from build_loss import build_loss_func
from dataset import return_dataset
from evaluator import evaluator
from models import build_model

SEED = 14
torch.manual_seed(SEED)


def get_arg():
    parser = argparse.ArgumentParser(description="image model for action segmentation")
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument("--device", type=int, default=-1)
    args = parser.parse_args()
    config = Dict(yaml.safe_load(open(args.config)))
    if args.device != -1:
        config.device = f"cuda:{str(args.device)}"
    return config


def sweep(path):
    config = dict(yaml.safe_load(open(path)))
    sweep_id = wandb.sweep(config, project="ActionPurposeSegmentation")
    wandb.agent(sweep_id, main)


def main(sweep=False):
    config = get_arg()

    config.save_folder = os.path.join(config.save_folder, config.model)
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)

    if config.wandb:
        name = config.model + "_" + config.head
        wandb.init(project="Asahi", config=config, name=name)
        config = wandb.config

    device = config.device
    print("device:", device)
    # train_set, test_set = return_data.return_dataset(config)
    train_set, test_set = return_dataset.return_dataloader(config)
    model = build_model.build_model(config)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = build_loss_func(config)

    best_eval = 0
    th = 0.3
    for epoch in range(1, 1 + config.epochs):
        print("\nepoch:", epoch)
        t0 = time.time()
        train_loss = train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataset=train_set,
            config=config,
            device=device,
        )
        scheduler.step()
        print(f"\nlr: {scheduler.get_last_lr()}")
        t1 = time.time()
        print(f"\ntraining time :{round(t1 - t0)} sec")

        best_eval = test(
            model=model,
            dataset=test_set,
            config=config,
            device=device,
            best_eval=best_eval,
            th=th,
        )
        if config.wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "loss": train_loss,
                    "acc": best_eval,
                    "lr": scheduler.get_last_lr(),
                }
            )
    path = os.path.join(config.save_folder, "model.pth")

    torch.save(model.state_dict(), path)


def train(model, optimizer, criterion, dataset, config, device):
    model.train()
    total_loss = 0
    counter = 0
    for data in Bar(dataset):
        label = data["label"]
        label = label.to(device)
        output = model(data["data"], device)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        counter += 1
    print(f"\rtotal_loss: [{total_loss / counter}]", end="")
    return total_loss / counter


def test(model, dataset, config, device, best_eval=0, th=0.6):
    model.eval()
    labels = []
    preds = []
    eval = evaluator()
    for data in Bar(dataset):
        # for img, label in dataset:
        # img = img.to(device)
        output = model(data["data"], device)["out"]
        # output = model(img)
        # output = torch.argmax(output, axis=1)
        output[output < th] = 0
        output[output >= th] = 1
        label = data["label"]

        labels.extend(label.detach().cpu().numpy())
        preds.extend(output.detach().cpu().numpy())

    labels, preds = np.array(labels), np.array(preds)
    # with open("test.pkl", "wb") as f:
    # pickle.dump(preds, f)
    eval.set_data(labels, preds)
    eval.print_eval(["accuracy"])
    score = eval.return_eval_score()
    if score > best_eval and score > th:
        path = os.path.join(
            config.save_folder,
            config.model
            + "_"
            + config.type
            + "_"
            + str(score).replace(".", "")
            + ".pth",
        )
        # torch.save(model.state_dict(), path)
    return max(score, best_eval)


if __name__ == "__main__":
    path = "config/config_tcn_sweep.yaml"
    main()
    # sweep(path)
