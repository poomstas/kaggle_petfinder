# %%
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
def print_config(config_dict):
    print('='*80)
    for key in config_dict.keys():
        print('{}:\t\t\t{}'.format(key, config_dict[key]))
    print('='*80)
# %%
def convert_data(image, label):
    image = to_pil_image(image).convert("RGB")
    label = label.detach().cpu().numpy()

    return image, label