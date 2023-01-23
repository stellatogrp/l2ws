import matplotlib.pyplot as plt
from pandas import read_csv
import sys
import jax.numpy as jnp
import pdb
import yaml
import os
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
import math
from utils.data_utils import recover_last_datetime
import imageio
from utils.data_utils import copy_data_file, recover_last_datetime
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    "font.size": 16,
})



def markowitz_gif(cfg):
    pass


def osc_mass_gif(cfg):
    pass


def vehicle_gif(cfg):
    pass

@hydra.main(config_path='configs/robust_pca', config_name='robust_pca_gif.yaml')
def robust_pca_gif(cfg):
    eval_iters_gif('robust_pca', cfg)


def get_data(example, datetime, eval_iters):
    orig_cwd = hydra.utils.get_original_cwd()
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/iters_compared.csv"
    df = read_csv(path)
    return df
    # if csv_title == 'last':
    #     last_column = df.iloc[:, -1]
    # else:
    #     last_column = df[csv_title]
    # return last_column[:eval_iters]


def get_k(orig_cwd, example, datetime):
    train_yaml_filename = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/.hydra/config.yaml"
    with open(train_yaml_filename, "r") as stream:
        try:
            out_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    k = int(out_dict['train_unrolls'])
    return k


def get_epochs_per_eval(orig_cwd, example, datetime):
    train_yaml_filename = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/.hydra/config.yaml"
    with open(train_yaml_filename, "r") as stream:
        try:
            out_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    eval_every_x_epochs = int(out_dict['eval_every_x_epochs'])
    return eval_every_x_epochs


def eval_iters_gif(example, cfg):
    '''
    first plot 
    '''
    orig_cwd = hydra.utils.get_original_cwd()
    datetimes = cfg.datetimes
    labels = cfg.labels
    if datetimes == '':
        # get the most recent datetime and update datetimes
        datetime = recover_last_datetime(orig_cwd, example, 'train')
        # cfg.datetime = datetime

        dt_cfg = {'datetime': datetime}

        with open('datetime.yaml', 'w') as file:
            yaml.dump(dt_cfg, file)

        datetimes = [datetime]

    if labels == '':
        labels = ['default' for i in range(len(datetimes))]

    
    data = np.zeros((len(datetimes), cfg.gif_length, cfg.eval_iters))
    for i in range(len(datetimes)):
        datetime = datetimes[i]
        df = get_data(example, datetime, cfg.eval_iters)

        exclude = ['final', 'iterations', 'Unnamed: 0', 'no_train', 'fixed_ws']
        no_learn = df['no_train']
        k = get_k(orig_cwd, example, datetime)
        if labels[i] == 'default':
            label = f"learned: k={k}"
        else:
            

        count = 0
        for col in df:
            if col not in exclude:
                if count < cfg.gif_length:
                    print('col', col)
                    data[i, count, :] = df[col]
                    count += 1
            # if count == 1:
            #     second_epochs_per_eval = int(col[11:])
    # assume that the evals_per_epoch is the same for all examples
    epochs_per_eval = get_epochs_per_eval(orig_cwd, example, datetimes[0])
    filenames = []

    for j in range(cfg.gif_length):
        for i in range(len(datetimes)):
            if cfg.add_k:
                label = f"{labels[i]}: {k}"
            else:
                label == labels[i]
            plt.plot(data[i, j, :], label=label)

        # assume no_learn comes from the first one
        plt.plot(no_learn, 'black', label='no learning')
        plt.ylim((cfg.y_low, cfg.y_high))
        plt.legend()
        filename = f"plot_{j}.png"
        filenames.append(filename)
        plt.yscale('log')
        epoch = epochs_per_eval * j
        plt.title(f"Epoch {epoch}")
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()

    # Build GIF
    with imageio.get_writer('eval_gif.gif', mode='I') as writer:
        for filename in filenames:  # ['1.png', '2.png', '3.png', '4.png']:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)


if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/gpfs/rajivs/learn2warmstart/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'markowitz':
        sys.argv[1] = base + 'markowitz/gifs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        markowitz_gif()
    elif sys.argv[1] == 'osc_mass':
        sys.argv[1] = base + 'osc_mass/gifs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        osc_mass_gif()
    elif sys.argv[1] == 'vehicle':
        sys.argv[1] = base + 'vehicle/gifs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        vehicle_gif()
    elif sys.argv[1] == 'robust_pca':
        sys.argv[1] = base + 'robust_pca/gifs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        robust_pca_gif()
