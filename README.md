# Intro

I worked on this project for just 6 weeks, so there are a lot of hard encoding, not refactoring, bugs. This repo is not completed.

# Install some library

```
pip install addict barbar wandb 
```

you need to install other libraryes if you have import error.

# Data Preparation

number of samples is so many. (100,000~)
It takes really much time to dataloading.

In thie repo, I made HDF5 data format for this data.
HDF5 is Hierarchical data structure, it uses dict-like data access fastly.
HDF5 data stracture is probably faster than normal data loading.

```
python dataset/mkhdf5.py 
```

it takes about 1 hour for training data parse.

mkhdf5.py generate hdf_dataset.h5 which include add data like image, label, path, data list as dict-like.
mkhdf5.py has a lot of hardencoding, so if you want, add argparse or something.
Please change some path.

# hdf5 Dataset

in asahi_changedet/dataset/asahi_dataset.py
There is CLASS Hdf5Dataset which uses hdf_dataset.h5.

# training

```
python train.py config/config_multi.yaml
```

I didnt search hyper parameters because I didnt have enough time.

# inference
Only evaluation and inference. Need model weight so run after training.

```
python inference config/config_multi.yaml
```
