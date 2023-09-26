import os

import hydra
import yaml


def recover_last_datetime(orig_cwd, example, stage):
    '''
    stage should be either
    1. data_setup
    2. aggregate
    3. train
    '''
    folder = f"{orig_cwd}/outputs/{example}/{stage}_outputs/"
        
    date_entries = os.listdir(folder)
    date_entries.sort()
    last_date = date_entries[len(date_entries)-1]
    date_folder = f"{folder}{last_date}"

    datetime_entries = os.listdir(date_folder)
    datetime_entries.sort()
    last_time = datetime_entries[len(datetime_entries)-1]

    last_datetime = f"{last_date}/{last_time}"
    return last_datetime

def copy_data_file(example, datetime):
    orig_cwd = hydra.utils.get_original_cwd()
    # data_yaml_filename = f"{orig_cwd}/outputs/{example}/aggregate_outputs/{datetime}/data_setup_copied.yaml"  # noqa
    data_yaml_filename = f"{orig_cwd}/outputs/{example}/data_setup_outputs/{datetime}/.hydra/config.yaml"  # noqa
    
    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # write the yaml file to the train_outputs folder
    with open('data_setup_copied.yaml', 'w') as file:
        yaml.dump(setup_cfg, file)
