from cProfile import label
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
from l2ws.utils.data_utils import recover_last_datetime
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    # "font.size": 24,
    "font.size": 16,
})


@hydra.main(config_path='configs/osc_mass', config_name='osc_mass_plot.yaml')
def osc_mass_plot_eval_iters(cfg):
    example = 'osc_mass'
    plot_eval_iters(example, cfg)


@hydra.main(config_path='configs/markowitz', config_name='markowitz_plot.yaml')
def markowitz_plot_eval_iters(cfg):
    example = 'markowitz'
    plot_eval_iters(example, cfg)


@hydra.main(config_path='configs/vehicle', config_name='vehicle_plot.yaml')
def vehicle_plot_eval_iters(cfg):
    example = 'vehicle'
    plot_eval_iters(example, cfg)

@hydra.main(config_path='configs/robust_pca', config_name='robust_pca_plot.yaml')
def robust_pca_plot_eval_iters(cfg):
    example = 'robust_pca'
    plot_eval_iters(example, cfg)


@hydra.main(config_path='configs/robust_kalman', config_name='robust_kalman_plot.yaml')
def robust_kalman_plot_eval_iters(cfg):
    example = 'robust_kalman'
    plot_eval_iters(example, cfg, train=False)
    overlay_training_losses(example, cfg)


@hydra.main(config_path='configs/robust_ls', config_name='robust_ls_plot.yaml')
def robust_ls_plot_eval_iters(cfg):
    example = 'robust_ls'
    overlay_training_losses(example, cfg)
    plot_eval_iters(example, cfg, train=False)
    
@hydra.main(config_path='configs/phase_retrieval', config_name='phase_retrieval_plot.yaml')
def phase_retrieval_plot_eval_iters(cfg):
    example = 'phase_retrieval'
    plot_eval_iters(example, cfg, train=False)

@hydra.main(config_path='configs/all', config_name='plot.yaml')
def plot_l4dc(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    examples = []
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=True)
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')

    # oscillating masses
    cfg_om = cfg.mpc
    om_nl = get_data('mpc', cfg_om.no_learning_datetime, 'no_train', cfg_om.eval_iters)
    om_nws = get_data('mpc', cfg_om.naive_ws_datetime, 'fixed_ws', cfg_om.eval_iters)
    axes[0].plot(om_nl, 'k-.')
    axes[0].plot(om_nws, 'm-.')
    example = 'mpc'
    for datetime in cfg_om.output_datetimes:
        # train_yaml_filename = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/.hydra/config.yaml"
        # with open(train_yaml_filename, "r") as stream:
        #     try:
        #         out_dict = yaml.safe_load(stream)
        #     except yaml.YAMLError as exc:
        #         print(exc)
        # k = int(out_dict['train_unrolls'])
        k = get_k(orig_cwd, example, datetime)
        curr_data = get_data(example, datetime, 'last', cfg_om.eval_iters)

        # plot
        axes[0].plot(curr_data)

    # vehicle
    cfg_ve = cfg.vehicle
    ve_nl = get_data('vehicle', cfg_ve.no_learning_datetime, 'no_train', cfg_ve.eval_iters)
    ve_nws = get_data('vehicle', cfg_ve.naive_ws_datetime, 'fixed_ws', cfg_ve.eval_iters)
    axes[1].plot(ve_nl, 'k-.')
    axes[1].plot(ve_nws, 'm-.')
    example = 'vehicle'
    for datetime in cfg_ve.output_datetimes:
        k = get_k(orig_cwd, example, datetime)
        curr_data = get_data(example, datetime, 'last', cfg_ve.eval_iters)

        # plot
        axes[1].plot(curr_data)

    # markowitz
    cfg_mark = cfg.markowitz
    mark_nl = get_data('markowitz', cfg_mark.no_learning_datetime, 'no_train', cfg_mark.eval_iters)
    mark_nws = get_data('markowitz', cfg_mark.naive_ws_datetime, 'fixed_ws', cfg_mark.eval_iters)
    axes[2].plot(mark_nl, 'k-.', label='no learning')
    axes[2].plot(mark_nws, 'm-.', label='nearest neighbor')
    example = 'markowitz'
    for datetime in cfg_mark.output_datetimes:
        k = get_k(orig_cwd, example, datetime)
        curr_data = get_data(example, datetime, 'last', cfg_mark.eval_iters)

        # plot
        axes[2].plot(curr_data, label=f"train $k={k}$")

    axes[2].legend()
    axes[0].set_xlabel('evaluation iterations')
    axes[1].set_xlabel('evaluation iterations')
    axes[2].set_xlabel('evaluation iterations')
    axes[0].set_ylabel('test fixed point residuals')
    axes[0].set_title('oscillating masses')
    axes[1].set_title('vehicle')
    axes[2].set_title('markowitz')

    plt.savefig('combined_plots.pdf', bbox_inches='tight')
    fig.tight_layout()


def get_k(orig_cwd, example, datetime):
    train_yaml_filename = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/.hydra/config.yaml"
    with open(train_yaml_filename, "r") as stream:
        try:
            out_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    k = int(out_dict['train_unrolls'])
    return k


def get_data(example, datetime, csv_title, eval_iters):
    orig_cwd = hydra.utils.get_original_cwd()
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/iters_compared.csv"
    df = read_csv(path)
    if csv_title == 'last':
        last_column = df.iloc[:, -1]
    else:
        last_column = df[csv_title]
    return last_column[:eval_iters]


def get_loss_data(example, datetime):
    orig_cwd = hydra.utils.get_original_cwd()

    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/train_test_results.csv"
    df = read_csv(path)
    # if csv_title == 'last':
    #     last_column = df.iloc[:, -1]
    # else:
    #     last_column = df[csv_title]
    # return last_column[:eval_iters]
    train_losses = df['train_loss']
    test_losses = df['test_loss']
    return train_losses, test_losses


def overlay_training_losses(example, cfg):
    orig_cwd = hydra.utils.get_original_cwd()

    # recover the datetimes
    datetimes = cfg.output_datetimes
    if datetimes == []:
        dt = recover_last_datetime(orig_cwd, example, 'train')
        datetimes = [dt]

    '''
    retrieve the training + test loss values held in 
    train_test_results.csv
    '''
    all_train_losses = []
    all_test_losses = []
    for i in range(len(datetimes)):
        datetime = datetimes[i]
        train_losses, test_losses = get_loss_data(example, datetime)
        all_train_losses.append(train_losses)
        all_test_losses.append(test_losses)

    # titles = cfg.loss_overlay_titles
    # for i in range(len(datetimes)):
    #     plt.plot(all_train_losses[i], label=f"train: {titles[i]}")
    #     plt.plot(all_test_losses[i], label=f"test: {titles[i]}")
    # plt.yscale('log')
    # plt.xlabel('epochs')
    # plt.ylabel('fixed point residual average')
    # plt.legend()
    # plt.savefig('losses_over_epochs.pdf', bbox_inches='tight')
    # plt.clf()

    # for i in range(len(datetimes)):
    #     plt.plot(all_train_losses[i], label=f"train: {titles[i]}")
    # plt.yscale('log')
    # plt.xlabel('epochs')
    # plt.ylabel('fixed point residual average')
    # plt.legend()
    # plt.savefig('train_losses_over_epochs.pdf', bbox_inches='tight')
    # plt.clf()

    """"
    now create table like

    k                   5       15      50  
    train reduction
    test reduction
    reduction gap
    """
    relative_loss_df = pd.DataFrame()
    relative_loss_df['rows'] = ['relative_train_loss', 'relative_test_loss', 'relative_gap']
    k_values = []
    rel_gen_gaps = []
    rel_tr_losses = []
    rel_te_losses = []
    for i in range(len(datetimes)):
        tr_losses = all_train_losses[i]
        te_losses = all_test_losses[i]
        orig_loss = te_losses[0]
        print(f"tr_losses", tr_losses)
        print(f"te_losses", te_losses)
        print(f"orig_loss: {orig_loss}")
        
        rel_train_loss = tr_losses[-1:].mean() / orig_loss
        rel_test_loss = te_losses.iloc[-1] / orig_loss
        rel_gap = rel_test_loss - rel_train_loss 
        k = get_k(orig_cwd, example, datetimes[i])
        k_values.append(k)
        col = f"k = {k}"
        row = np.array([rel_train_loss, rel_test_loss, rel_gap])
        rel_gen_gaps.append(rel_gap)
        rel_tr_losses.append(rel_train_loss)
        rel_te_losses.append(rel_test_loss)
        print(f"row: {row}")
        relative_loss_df[col] = np.round(row, decimals=3)
    relative_loss_df.to_csv('relative_losses.csv')

    # now give a plot for the generalization
    plt.plot(np.array(k_values), np.array(rel_gen_gaps))
    plt.xlabel('k')
    plt.ylabel('relative generalization gap')
    plt.savefig('relative_generalization_gaps.pdf')
    plt.clf()

    # plot the relative train and test final losses
    # now give a plot for the generalization
    plt.plot(np.array(k_values), np.array(rel_tr_losses), label='train')
    plt.plot(np.array(k_values), np.array(rel_te_losses), label='test')
    plt.xlabel('k')
    plt.ylabel('relative final losses')
    plt.legend()
    plt.savefig('relative_final_losses.pdf')
    plt.clf()



def plot_eval_iters(example, cfg, train=False):
    '''
    get the datetimes
    1. no learning
    2. list of fully trained models
    3. pretraining only
    '''
    orig_cwd = hydra.utils.get_original_cwd()
    eval_iters = cfg.eval_iters

    datetimes = cfg.output_datetimes
    if datetimes == []:
        dt = recover_last_datetime(orig_cwd, example, 'train')
        datetimes = [dt]

    pretrain_datetime = cfg.pretrain_datetime

    no_learning_datetime = cfg.no_learning_datetime
    if no_learning_datetime == '':
        no_learning_datetime = recover_last_datetime(orig_cwd, example, 'train')

    naive_ws_datetime = cfg.naive_ws_datetime
    if naive_ws_datetime == '':
        naive_ws_datetime = recover_last_datetime(orig_cwd, example, 'train')

    accs = cfg.accuracies
    df_acc = pd.DataFrame()
    df_acc['accuracies'] = np.array(accs)

    if train:
        iters_file = "iters_compared_train.csv"
    else:
        iters_file = "iters_compared_test.csv"

    '''
    no learning
    '''
    no_learning_path = f"{orig_cwd}/outputs/{example}/train_outputs/{no_learning_datetime}/{iters_file}"
    no_learning_df = read_csv(no_learning_path)
    last_column = no_learning_df['no_train']
    plt.plot(last_column[:eval_iters], 'k-.', label='no learning')
    second_derivs_no_learn = second_derivative_fn(np.log(last_column[:eval_iters]))
    df_acc = update_acc(df_acc, accs, 'no_learn', last_column[:eval_iters])

    '''
    naive warm start
    '''
    naive_ws_path = f"{orig_cwd}/outputs/{example}/train_outputs/{naive_ws_datetime}/{iters_file}"
    naive_ws_df = read_csv(naive_ws_path)
    # last_column = naive_ws_df['fixed_ws']
    last_column = naive_ws_df['nearest_neighbor']
    # plt.plot(last_column[:eval_iters], 'm-.', label='naive warm start')
    plt.plot(last_column[:eval_iters], 'm-.', label='nearest neighbor')
    second_derivs_naive_ws = second_derivative_fn(np.log(last_column[:eval_iters]))
    df_acc = update_acc(df_acc, accs, 'naive_ws', last_column[:eval_iters])

    '''
    pretraining
    '''
    if pretrain_datetime != '':
        pretrain_path = f"{orig_cwd}/outputs/{example}/train_outputs/{pretrain_datetime}/{iters_file}"
        pretrain_df = read_csv(pretrain_path)
        last_column = pretrain_df['pretrain']
        plt.plot(last_column[:eval_iters], 'r+', label='pretrain')

    k_vals = np.zeros(len(datetimes))
    second_derivs = []
    titles = cfg.loss_overlay_titles
    for i in range(len(datetimes)):
        datetime = datetimes[i]
        path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/{iters_file}"
        df = read_csv(path)

        '''
        for the fully trained models, track the k value
        - to do this, load the train_yaml file
        '''
        train_yaml_filename = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/.hydra/config.yaml"
        with open(train_yaml_filename, "r") as stream:
            try:
                out_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        k = out_dict['train_unrolls']
        k_vals[i] = k

        last_column = df.iloc[:, -1]
        second_derivs.append(second_derivative_fn(np.log(last_column[:eval_iters])))
        # plt.plot(last_column[:250], label=f"train k={k}")
        if titles == []:
            plt.plot(last_column[:eval_iters], label=f"train $k={int(k_vals[i])}$")
        else:
            plt.plot(last_column[:eval_iters], label=f"train $k={int(k_vals[i])}$, {titles[i]}")
        df_acc = update_acc(df_acc, accs, f"traink{int(k_vals[i])}", last_column[:eval_iters])

    plt.yscale('log')
    plt.xlabel('evaluation iterations')
    plt.ylabel('test fixed point residuals')
    plt.legend()
    plt.savefig('eval_iters.pdf', bbox_inches='tight')
    plt.clf()


    '''
    save the iterations required to reach a certain accuracy
    '''
    df_acc.to_csv('accuracies.csv')
    df_percent = pd.DataFrame()
    df_percent['accuracies'] = np.array(accs)
    no_learning_acc = df_acc['no_learn']
    for col in df_acc.columns:
        if col != 'accuracies':
            val = 1 - df_acc[col] / no_learning_acc
            df_percent[col] = np.round(val, decimals=2)
    
    df_percent.to_csv('iteration_reduction.csv')

    '''
    save both iterations and fraction reduction in single table
    '''
    df_acc_both = pd.DataFrame()
    df_acc_both['accuracies'] = df_acc['no_learn']
    df_acc_both['no_learn_iters'] = np.array(accs)

    for col in df_percent.columns:
        if col != 'accuracies' and col != 'no_learn':
            df_acc_both[col + '_iters'] = df_acc[col]
            df_acc_both[col + '_red'] = df_percent[col]
    df_acc_both.to_csv('accuracies_reduction_both.csv')

    '''
    now plot the 2nd derivative of the evaluation iterations

    plot 1: 2nd derivative of all runs
    plot 2: train_unrolls vs max_curvature
        for plot 2: ignore no-learning, pretraining
    '''
    # plot 1
    plt.plot(second_derivs_no_learn[5:], label=f"no learning")
    if pretrain_datetime != '':
        plt.plot(second_derivs_pretrain, label=f"pretraining")

    max_second_derivs = np.zeros(len(datetimes))
    for i in range(len(datetimes)):
        cutoff = 5
        if k_vals[i] > 10:
            cutoff = 15
        max_second_derivs[i] = np.argmax(second_derivs[i][cutoff:]) + cutoff
        plt.plot(second_derivs[i][5:], label=f"train $k={k_vals[i]}$")
    
    plt.legend()
    plt.savefig('second_derivatives.pdf', bbox_inches='tight')
    plt.clf()

    # plot 2 
    plt.scatter(k_vals, max_second_derivs)
    xx = np.arange(k_vals.max())
    plt.plot(xx)
    plt.xlabel('train iterations')
    plt.ylabel('maximum curvature iterations')
    plt.xlim([0, k_vals.max()+5])
    plt.ylim([0, k_vals.max()+5])
    plt.legend()
    plt.savefig('max_curvature.pdf', bbox_inches='tight')
    plt.clf()
    print('2nd deriv', second_derivs[-1])

    # first deriv
    data = last_column[:eval_iters]
    log_data = np.log(data)
    box = np.ones(20)/20
    smooth_data = np.convolve(log_data, box, mode='valid')
    deriv1 = np.diff(smooth_data)
    deriv2 = np.diff(deriv1)
    plt.plot(deriv1)
    plt.plot(deriv2)
    plt.savefig('first_deriv.pdf', bbox_inches='tight')
    plt.clf()

    plt.plot(smooth_data)
    plt.savefig('smooth.pdf', bbox_inches='tight')
    plt.clf()

    smooth_deriv1 = np.convolve(deriv1, box, mode='valid')
    smooth_deriv2 = np.convolve(deriv2, box, mode='valid')
    plt.plot(smooth_deriv1)
    plt.plot(smooth_deriv2)
    plt.savefig('smooth_deriv_plots.pdf', bbox_inches='tight')
    plt.clf()

    '''
    now write the iterations required to reach certain accuracy
    '''


def update_acc(df_acc, accs, col, losses):
    iter_vals = np.zeros(len(accs))
    for i in range(len(accs)):
        if losses.min() < accs[i]:
            iter_vals[i] = int(np.argmax(losses < accs[i]))
        else:
            iter_vals[i] = losses.size
    int_iter_vals = iter_vals.astype(int)
    df_acc[col] = int_iter_vals
    return df_acc


def second_derivative_fn(x):
    dydx = np.diff(x)
    dy2d2x = np.diff(dydx)
    return dy2d2x


if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/gpfs/rajivs/learn2warmstart/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'markowitz':
        sys.argv[1] = base + 'markowitz/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        markowitz_plot_eval_iters()
    elif sys.argv[1] == 'osc_mass':
        sys.argv[1] = base + 'osc_mass/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        osc_mass_plot_eval_iters()
    elif sys.argv[1] == 'vehicle':
        sys.argv[1] = base + 'vehicle/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        vehicle_plot_eval_iters()
    elif sys.argv[1] == 'robust_kalman':
        sys.argv[1] = base + 'robust_kalman/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        robust_kalman_plot_eval_iters()
    elif sys.argv[1] == 'robust_pca':
        sys.argv[1] = base + 'robust_pca/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        robust_pca_plot_eval_iters()
    elif sys.argv[1] == 'robust_ls':
        sys.argv[1] = base + 'robust_ls/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        robust_ls_plot_eval_iters()
    elif sys.argv[1] == 'phase_retrieval':
        sys.argv[1] = base + 'phase_retrieval/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        phase_retrieval_plot_eval_iters()
    elif sys.argv[1] == 'all':
        sys.argv[1] = base + 'all/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        plot_l4dc()
