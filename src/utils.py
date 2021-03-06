# %%
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from torchvision.transforms.functional import to_pil_image

# %%
def print_config(config_dict):
    print('='*80)
    for key in config_dict.keys():
        tab_spacings = '\t\t\t' if len(key)<=6 else '\t\t'
        print('{}:{}{}'.format(key, tab_spacings, config_dict[key]))
    print('='*80)

# %%
def preprocess_data(csv_path, val_frac, abridge_frac=1.0, scale_target=True, random_state=12345):
    df = pd.read_csv(csv_path)

    if abridge_frac != 1.0:
        print('Total dataset abridged by a factor of {:.2f} before splitting.'.format(abridge_frac))
        df = df.sample(frac=abridge_frac, replace=False)
    
    if scale_target:
        df['Pawpularity'] /= 100.0

    n_val = int(len(df) * val_frac)
    df_train, df_val = train_test_split(df, test_size=n_val, random_state=random_state)

    print(df_train)
    print(df_val)
    print('Saving to separate files...')
    print('Writing to: ./data/separated_train.csv')
    print('Writing to: ./data/separated_val.csv')

    df_train.to_csv('./data/separated_train.csv', index=False)
    df_val.to_csv('./data/separated_val.csv', index=False)

    return df_train, df_val

# %%
def get_writer_name(config):
    writer_name = \
        "PetFindr_{}_LR_{}_BS_{}_nEpoch_{}_{}".format(
            config['model'], config['lr'], config['batch_size'], config['epochs'], 
            datetime.now().strftime("%Y%m%d_%H%M%S"))

    if config['note'] != "":
        TB_text = config['note'].replace(' ', '_') # Replace spaces with underscore
        writer_name += "_" + TB_text

    print('Case Name: {}'.format(writer_name))

    return writer_name

# %%
def parse_arguments(parser):
    parser.add_argument('--gpu_index',      type=int,   default=0,              help='GPU Index, default at 0')
    parser.add_argument('--model',          type=str,   default='xceptionimg',  help='Backbone Model')
    parser.add_argument('--batch_size',     type=int,   default=32,             help='Batch Size')
    parser.add_argument('--drop_last',      type=bool,  default=False,          help='Drop last mismatched batch')
    parser.add_argument('--train_shuffle',  type=bool,  default=True,           help='Shuffle training data')
    parser.add_argument('--val_shuffle',    type=bool,  default=False,          help='Shuffle validation data')
    parser.add_argument('--num_workers',    type=int,   default=1,              help='Number of workers for DataLoader')
    parser.add_argument('--lr',             type=float, default=0.001,          help='Learning rate')
    parser.add_argument('--lr_min',         type=float, default=1e-10,          help='Minimum bounds for reducing learning rate')
    parser.add_argument('--lr_patience',    type=int,   default=5,              help='Patience for learning rate plateau detection')
    parser.add_argument('--lr_reduction',   type=float, default=0.1,            help='Learning rate reduction factor in case of plateau')
    parser.add_argument('--abridge_frac',   type=float, default=0.1,            help='Fraction of the original training data to be used for train+val')
    parser.add_argument('--val_frac',       type=float, default=0.1,            help='Fraction of the training data (abridged or not) to be used for validation set')
    parser.add_argument('--scale_target',   type=bool,  default=True,           help='Scale Pawpularity from 0-100 to 0-1')
    parser.add_argument('--epochs',         type=int,   default=30,             help='Total number of epochs to train over')
    parser.add_argument('--note',           type=str,   default="",             help='Note to leave on TensorBoard and W&B')

    args = parser.parse_args()

    return args

# %%
def get_dict_from_args(args):
    params = [item for item in dir(args) if not item.startswith('_')]
    config = {}
    for item in params:
        config[item] = getattr(args, item)

    return config

# %%
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training Hyperparameters')
    args = parse_arguments(parser)

    config = get_dict_from_args(args)

