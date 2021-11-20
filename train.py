# %%
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
DATA_DIR = './data/'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

df_train = pd.read_csv('./data/train_csv')
df_test  = pd.read_csv('./data/test_csv')

# %%

# %%
