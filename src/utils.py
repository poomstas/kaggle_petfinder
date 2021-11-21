# %%
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from torchvision.transforms.functional import to_pil_image

# %%
def get_writer_name(args, config):
    writer_name = \
        "TB_NPHD2021_{}_LR_{}_BS_{}_nEpoch_{}".format(
            config.model, config.lr, config.batch_size, config.epochs, 
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    if args.TB_note != "":
        writer_name += "_" + config.TB_note

    print('TensorBoard Name: {}'.format(writer_name))

    return writer_name

# %%
def separate_train_val(csv_path, val_frac, random_state=12345):
    df = pd.read_csv(csv_path)
    n_val = int(len(df) * val_frac)
    df_train, df_val = train_test_split(df, test_size=n_val, random_state=random_state)

    df_train = df_train.reset_index()
    df_val = df_val.reset_index()

    print(df_train)
    print(df_val)
    print('Saving to separate files...')
    print('Writing to: ./data/separated_train.csv')
    print('Writing to: ./data/separated_val.csv')
    df_train.to_csv('./data/separated_train.csv')
    df_val.to_csv('./data/separated_val.csv')

    return df_train, df_val

# %%
def print_config(config_dict):
    print('='*80)
    for key in config_dict.keys():
        tab_spacings = '\t\t\t' if len(key)<=6 else '\t\t'
        print('{}:{}{}'.format(key, tab_spacings, config_dict[key]))
    print('='*80)
# %%
def convert_data(image, label):
    image = to_pil_image(image).convert("RGB")
    label = label.detach().cpu().numpy()

    return image, label
# %%
if __name__=='__main__':
    print(os.getcwd())
    csv_path = '../data/train.csv'
    separate_train_val(csv_path=csv_path, val_frac=0.1)
