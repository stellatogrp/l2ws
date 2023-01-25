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
    angle_prob_nums = cfg.angle_prob_nums
    for prob_num in angle_prob_nums:
        angles_gif('robust_pca', cfg, prob_num)
    # if cfg.gradient:
    #     eval_iters_gif('robust_pca', cfg, gradient=True)
    # eval_iters_gif('robust_pca', cfg)



def get_data(example, datetime, eval_iters):
    orig_cwd = hydra.utils.get_original_cwd()
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/iters_compared.csv"
    df = read_csv(path)
    return df


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


def angles_gif(example, cfg, prob_num):
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

    data = np.zeros((len(datetimes), cfg.gif_length, cfg.eval_iters - 3))
    ks = []
    for i in range(len(datetimes)):
        datetime = datetimes[i]
        df = get_data(example, datetime, cfg.eval_iters)

        exclude = ['final', 'iterations', 'Unnamed: 0', 'no_train', 'fixed_ws']
        # no_learn = df['no_train']

        path_no_learn = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/polar/no_train/angle_data.csv"
        angle_df_no_learn = read_csv(path_no_learn)
        no_learn = np.array(angle_df_no_learn.iloc[prob_num, 1:-1])

        k = get_k(orig_cwd, example, datetime)
        ks.append(k)

        count = 0
        for col in df:
            if col not in exclude:
                '''
                don't use that df for eval_iters
                use the df from polar/{col}/angle_data.csv
                '''

                path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/polar/{col}/angle_data.csv"
                angle_df = read_csv(path)

                if count < cfg.gif_length:
                    print('col', col)
                    data[i, count, :] = np.array(angle_df.iloc[prob_num, 1:-1])
                    count += 1

    # assume that the evals_per_epoch is the same for all examples
    epochs_per_eval = get_epochs_per_eval(orig_cwd, example, datetimes[0])
    filenames = []

    y_low = data.min()
    y_high = data.max()
    print('y_low', y_low)
    print('y_high', y_high)
    # pdb.set_trace()
    for j in range(cfg.gif_length):
        for i in range(len(datetimes)):
            print('j', j)
            datetime = datetimes[i]
            
            if labels[i] == 'default':
                label = f"learned: k={ks[i]}"
            else:
                if cfg.add_k:
                    label = f"{labels[i]}: {ks[i]}"
                else:
                    label == labels[i]
            plt.plot(data[i, j, :], label=label)

        # assume no_learn comes from the first one
        plt.plot(no_learn, 'black', label='no learning')
        plt.ylim((y_low, y_high))
        plt.legend()
        filename = f"plot_{j}.png"
        filenames.append(filename)
        # plt.yscale('log')
        epoch = epochs_per_eval * j
        plt.title(f"Problem {prob_num} angles: Epoch {epoch}")
        plt.ylabel('angle')
        plt.xlabel('evaluation steps')
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()

    # Build GIF
    with imageio.get_writer(f"angles_gif_{prob_num}.gif", mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)


def eval_iters_gif(example, cfg, gradient=False):
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

    if gradient:
        data = np.zeros((len(datetimes), cfg.gif_length, cfg.eval_iters - 1))
    else:
        data = np.zeros((len(datetimes), cfg.gif_length, cfg.eval_iters))
    ks = []
    for i in range(len(datetimes)):
        datetime = datetimes[i]
        df = get_data(example, datetime, cfg.eval_iters)

        exclude = ['final', 'iterations', 'Unnamed: 0', 'no_train', 'fixed_ws']
        no_learn = df['no_train']

        k = get_k(orig_cwd, example, datetime)
        ks.append(k)
        
        count = 0
        for col in df:
            if col not in exclude:
                if count < cfg.gif_length:
                    print('col', col)
                    if gradient:
                        data[i, count, :] = -np.diff(np.log10(df[col]))
                    else:
                        data[i, count, :] = df[col]
                    count += 1
            # if count == 1:
            #     second_epochs_per_eval = int(col[11:])
    # assume that the evals_per_epoch is the same for all examples
    epochs_per_eval = get_epochs_per_eval(orig_cwd, example, datetimes[0])
    filenames = []

    y_low = data.min()
    y_high = data.max()
    print('y_low', y_low)
    print('y_high', y_high)
    for j in range(cfg.gif_length):
        for i in range(len(datetimes)):
            datetime = datetimes[i]
            
            if labels[i] == 'default':
                label = f"learned: k={ks[i]}"
            else:
                if cfg.add_k:
                    label = f"{labels[i]}: {ks[i]}"
                else:
                    label == labels[i]
            plt.plot(data[i, j, :], label=label)

        # assume no_learn comes from the first one
        if gradient:
            plt.plot(-np.diff(np.log10(no_learn)), 'black', label='no learning')
        else:
            plt.plot(no_learn, 'black', label='no learning')
        # plt.ylim((cfg.y_low, cfg.y_high))
        plt.ylim((y_low, y_high))
        plt.legend()
        filename = f"plot_{j}.png"
        filenames.append(filename)
        plt.yscale('log')
        epoch = epochs_per_eval * j
        if gradient:
            plt.title(f"Gradient evals: Epoch {epoch}")
        else:
            plt.title(f"Epoch {epoch}")
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()

    # Build GIF
    if gradient:
        gif_name = 'grad_eval_gif.gif'
    else:
        gif_name = 'eval_gif.gif'
    with imageio.get_writer(gif_name, mode='I') as writer:
        for filename in filenames:
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
