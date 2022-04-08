from .asahi_dataset import AsahiDataset, Hdf5Dataset
from .transform import return_img_transform

from torch.utils.data import DataLoader
import torch


def return_dataset(config, fold_index=0):
    dataset_type = config.type
    csv_dir = config.csv_dir
    transforms = return_img_transform()
    hdf5 = True
    if hdf5:
        train_dataset = Hdf5Dataset(
            mode="train",
            transform=transforms,
            fold_index=fold_index,
            img_type=config.img_type,
            split_dataset=config.split_dataset,
        )
        test_dataset = Hdf5Dataset(
            mode="test",
            transform=transforms,
            fold_index=fold_index,
            img_type=config.img_type,
            split_dataset=config.split_dataset,
        )
    else:
        train_dataset = AsahiDataset(
            mode="train", transform=transforms, fold_index=fold_index, csv_dir=csv_dir
        )
        test_dataset = AsahiDataset(
            mode="test", transform=transforms, fold_index=fold_index, csv_dir=csv_dir
        )

    return train_dataset, test_dataset


def return_dataloader(config):
    train_set, test_set = return_dataset(config)
    n_worker = 8
    train_loader = DataLoader(
        train_set,
        batch_size=config.batchsize,
        shuffle=True,
        drop_last=True,
        num_workers=n_worker,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batchsize,
        drop_last=True,
        num_workers=8,
    )
    return train_loader, test_loader
