# %%
import os
import pandas as pd
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
def separate_train_val(csv_path):
    df = pd.read_csv(csv_path)
    print(df)

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
    separate_train_val(csv_path=csv_path)
