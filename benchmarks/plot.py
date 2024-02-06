import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pandas import read_csv

from l2ws.utils.data_utils import recover_last_datetime

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    "font.size": 25,
    # "font.size": 16,
})
cmap = plt.cm.Set1
colors = cmap.colors
titles_2_colors = dict(cold_start='black', 
                       nearest_neighbor=colors[6], 
                       prev_sol=colors[4],
                       reg_k0=colors[3],
                       reg_k2=colors[0],
                       reg_k5=colors[0],
                       reg_k5_deterministic=colors[2],
                       reg_k10=colors[0],
                       reg_k10_deterministic=colors[2],
                       reg_k15=colors[0],
                       reg_k15_deterministic=colors[2],
                       reg_k30=colors[5],
                       reg_k60=colors[2],
                    #    reg_k120=colors[0],
                       obj_k0=colors[3],
                       obj_k5=colors[0],
                       obj_k15=colors[0],
                       obj_k15_deterministic=colors[2],
                       obj_k30=colors[5],
                       obj_k60=colors[2])
                    #    obj_k120='gray')
titles_2_styles = dict(cold_start='-.', 
                       nearest_neighbor='-.', 
                       prev_sol='-.',
                       reg_k0='-',
                       reg_k2='-',
                       reg_k5='-',
                       reg_k5_deterministic='-',
                       reg_k10='-',
                       reg_k10_deterministic='-',
                       reg_k15='-',
                       reg_k15_deterministic='-',
                       reg_k30='-',
                       reg_k60='-',
                       reg_k120='-',
                       obj_k0='-',
                       obj_k5='-',
                       obj_k15='-',
                       obj_k15_deterministic='-',
                       obj_k30='-',
                       obj_k60='-')
                    #    obj_k120='-')
titles_2_markers = dict(cold_start='v', 
                       nearest_neighbor='<', 
                       prev_sol='^',
                       reg_k0='>',
                       reg_k2='o',
                       reg_k5='o',
                       reg_k5_deterministic='D',
                       reg_k10='o',
                       reg_k10_deterministic='D',
                       reg_k15='o',
                       reg_k15_deterministic='D',
                       reg_k30='x',
                       reg_k60='D',
                    #    reg_k120='-',
                       obj_k0='>',
                       obj_k5='o',
                       obj_k15='o',
                       obj_k15_deterministic='D',
                       obj_k30='x',
                       obj_k60='D')
titles_2_marker_starts = dict(cold_start=0, 
                       nearest_neighbor=16, 
                       prev_sol=23,
                       reg_k0=8,
                       reg_k2=4,
                       reg_k5=4,
                       reg_k10=4,
                       reg_k15=12,
                       reg_k30=0,
                       reg_k60=20,
                    #    reg_k120='-',
                       obj_k0=8,
                       obj_k5=4,
                       obj_k15=12,
                       obj_k30=0,
                       obj_k60=20)
                    #    obj_k120='-')
# titles_2_colors = dict(cold_start='black', 
#                        nearest_neighbor='magenta', 
#                        prev_sol='cyan',
#                        reg_k0=mcolors.TABLEAU_COLORS['tab:brown'],
#                        reg_k5='blue',
#                        reg_k15='red',
#                        reg_k30='green',
#                        reg_k60='orange',
#                        reg_k120='gray',
#                        obj_k0=mcolors.TABLEAU_COLORS['tab:brown'],
#                        obj_k5='blue',
#                        obj_k15='red',
#                        obj_k30='green',
#                        obj_k60='orange',
#                        obj_k120='gray')
# titles_2_styles = dict(cold_start='-.', 
#                        nearest_neighbor='-.', 
#                        prev_sol='-.',
#                        reg_k0='-',
#                        reg_k5='-',
#                        reg_k15='-',
#                        reg_k30='-',
#                        reg_k60='-',
#                        reg_k120='-',
#                        obj_k0='-',
#                        obj_k5='-',
#                        obj_k15='-',
#                        obj_k30='-',
#                        obj_k60='-',
#                        obj_k120='-')


@hydra.main(config_path='configs/sparse_coding', config_name='sparse_coding_plot.yaml')
def sparse_coding_plot_eval_iters(cfg):
    example = 'sparse_coding'
    # overlay_training_losses(example, cfg)
    # create_journal_results(example, cfg, train=False)
    # create_genL2O_results()
    metrics, timing_data, titles = get_all_data(example, cfg, train=False)
    cmap = plt.cm.Set1
    colors = cmap.colors
    colors = [colors[i] for i in range(len(colors))]
    colors[5] = colors[7]
    markers = ['o', 's', '>', '^', 'D', 'X']

    if len(titles) == 4:
        titles[-2] = titles[-2] + '_deterministic'
    nmse = metrics[0]
    for i in range(len(nmse)):
        # if titles[i] != 'cold_start' and titles[i] != 'nearest_neighbor':
        #     plt.plot(nmse[i])
        if titles[i] == 'cold_start' or titles[i] == 'nearest_neighbor':
            plt.plot(nmse[i][:cfg.eval_iters],
                    linestyle=titles_2_styles[titles[i]], 
                    color=titles_2_colors[titles[i]],
                    # color=colors[2*i-2],
                    marker=titles_2_markers[titles[i]],
                    #  markevery=(0, 2)
                    )
        else:
            plt.plot(nmse[i][:cfg.eval_iters],
                    linestyle=titles_2_styles[titles[i]], 
                    # color=titles_2_colors[titles[i]],
                    color=colors[2*(i-2)],
                    marker=markers[2*(i-2)],
                    #  markevery=(0, 2)
                    )
    plt.tight_layout()
    plt.xlabel('evaluation steps')
    plt.ylabel("NMSE (dB)")
    plt.savefig('nmse.pdf', bbox_inches='tight')
    plt.clf()

    # get frac_solved data
    #   for each acc
    #       for each datetime provided, 
    #           get the test frac solved
    #           get the pac_bayes bound (train + penalty)
    #           plot it
    out = get_frac_solved_data(example, cfg)
    all_test_results, all_pac_bayes_results, cold_start_results, nearest_neighbor_results = out
    # markers = ['o', 's', '>', '^', 'D', 'X']
    
    # import pdb
    # pdb.set_trace()
    styles = ['-', '-']
    for i in range(len(cfg.accuracies)):
        # plot ista and fista
        mark_start = titles_2_marker_starts['cold_start']
        plt.plot(cold_start_results[i][:cfg.eval_iters], 
                 linestyle=titles_2_styles['cold_start'], 
                 color=titles_2_colors['cold_start'],
                 marker=titles_2_markers['cold_start'],
                 markevery=(0, 2)
                 )
        mark_start = titles_2_marker_starts['nearest_neighbor']
        plt.plot(nearest_neighbor_results[i][:cfg.eval_iters], 
                 linestyle=titles_2_styles['nearest_neighbor'], 
                 color=titles_2_colors['nearest_neighbor'],
                 marker=titles_2_markers['nearest_neighbor'],
                 markevery=(1, 2))

        # plot the learned variants
        acc = cfg.accuracies[i]
        curr_test_results = all_test_results[i]
        curr_pac_bayes_results = all_pac_bayes_results[i]
        for j in range(len(curr_test_results)):
            plt.plot(curr_test_results[j], 
                     linestyle='-', 
                     color=colors[0 + 2 * j], 
                     marker=markers[0 + 2 * j],
                     markevery=(0, 2))
            plt.plot(curr_pac_bayes_results[j], 
                     linestyle='-', 
                     color=colors[1 + 2* j], 
                     marker=markers[1 + 2 * j],
                     markevery=(1, 2))
        plt.tight_layout()
        plt.xlabel('evaluation steps')
        plt.ylabel(f"frac. at {acc} NMSE (dB)")
        plt.savefig(f"acc_{acc}.pdf", bbox_inches='tight')
        plt.clf()
    # import pdb
    # pdb.set_trace()

    plot_conv_rates(example, cfg)


def plot_conv_rates(example, cfg):
    # getting curve data

    # plot the pac_bayes curve and the test curve
    out = get_conv_rates_data(example, cfg)
    conv_rates, all_test_results, all_pac_bayes_results, cold_start_results, nearest_neighbor_results = out
    markers = ['o', 's', '>', '^', 'o', 's']
    cmap = plt.cm.Set1
    colors = cmap.colors
    colors = [colors[i] for i in range(len(colors))]
    colors[5] = colors[7]
    styles = ['-', '-']
    # for i in range(len(cfg.accuracies)):
    # plot ista and fista
    mark_start = titles_2_marker_starts['cold_start']

    

    plt.plot(conv_rates,
        cold_start_results, 
                linestyle=titles_2_styles['cold_start'], 
                color=titles_2_colors['cold_start'],
                marker=titles_2_markers['cold_start'],
                # markevery=(30, 100)
                )
    mark_start = titles_2_marker_starts['nearest_neighbor']
    plt.plot(conv_rates,
        nearest_neighbor_results, 
                linestyle=titles_2_styles['nearest_neighbor'], 
                color=titles_2_colors['nearest_neighbor'],
                marker=titles_2_markers['nearest_neighbor'],
                # markevery=(60, 100)
                )

    # plot the learned variants
    # acc = cfg.accuracies[i]
    curr_test_results = all_test_results
    curr_pac_bayes_results = all_pac_bayes_results
    for j in range(len(curr_test_results)):
        plt.plot(conv_rates, curr_test_results[j][:conv_rates.size], 
                    linestyle='-', 
                    color=colors[0 + 2 * j], 
                    marker=markers[0 + 2 * j],
                    # markevery=(80, 100)
                    )
        plt.plot(conv_rates, curr_pac_bayes_results[j][:conv_rates.size], 
                    linestyle='-', 
                    color=colors[1 + 2 * j], 
                    # markevery=(0, 100),
                    marker=markers[1])
    plt.tight_layout()
    plt.xlabel('convergence rates')
    plt.ylabel("frac. of steps satisfied")
    plt.savefig("conv_rates.pdf", bbox_inches='tight')
    plt.clf()

    import pdb
    pdb.set_trace()


def create_classical_results(example, cfg):
    # example = 'sparse_coding'
    # overlay_training_losses(example, cfg)
    # create_journal_results(example, cfg, train=False)
    # create_genL2O_results()
    metrics, timing_data, titles = get_all_data(example, cfg, train=False)
    eval_iters = int(cfg.eval_iters)

    if len(titles) == 4:
        titles[-2] = titles[-2] + '_deterministic'
    nmse = metrics[0]
    for i in range(len(nmse)):
        # if titles[i] != 'cold_start' and titles[i] != 'nearest_neighbor':
        #     plt.plot(nmse[i])
        plt.plot(nmse[i][:eval_iters],
                 linestyle=titles_2_styles[titles[i]], 
                 color=titles_2_colors[titles[i]],
                 marker=titles_2_markers[titles[i]],
                 markevery=(0, 100)
                 )
    plt.tight_layout()
    plt.xlabel('evaluation steps')
    plt.ylabel("fixed-point residual")
    plt.yscale('log')
    plt.savefig('fp_res.pdf', bbox_inches='tight')
    plt.clf()


    cold_start_results, guarantee_results = get_frac_solved_data_classical(example, cfg)
    

    # worst case
    # worst_case = np.zeros(eval_iters)
    # steps = np.arange(cfg.eval_iters)
    # z_star_max = # Specify the CSV file name

    steps1 = np.arange(cold_start_results[i].size)
    steps2 = np.linspace(cold_start_results[i].size, eval_iters, 100000)
    steps = np.concatenate([steps1, steps2])


    z_star_max, theta_max = get_worst_case_datetime(example, cfg)

    markers = ['o', 's', '<', 'D']
    cmap = plt.cm.Set1
    colors = cmap.colors
    styles = ['-', '-']
    for i in range(len(cfg.accuracies)):
        acc = cfg.accuracies[i]
        mark_start = titles_2_marker_starts['cold_start']
        if cfg.worst_case:
            curr_curve = np.zeros(steps.size)
            curr_size = cold_start_results[i].size

            # prob bounds
            curr_curve[:curr_size] = cold_start_results[i] #[:cfg.eval_iters]

            # worst-case bounds
            indices = 1 / np.sqrt(steps + 2) * z_star_max * 1.1 < acc

            curr_curve[curr_size:] = cold_start_results[i].max()

            curr_curve[indices] = 1.0
        else:
            curr_curve = cold_start_results[i][:eval_iters]
        plt.plot(steps,
                curr_curve, 
                linestyle=titles_2_styles['cold_start'], 
                color=titles_2_colors['cold_start'],
                marker=titles_2_markers['cold_start'],
                linewidth=2.0,
                # markevery=(30, 100)
                markevery=(0.05, 0.1)
                )
        worst_case_curve = np.zeros(steps.size)
        worst_case_curve[indices] = 1.0

        plt.plot(steps,
                worst_case_curve, 
                linestyle=titles_2_styles['nearest_neighbor'], 
                color=titles_2_colors['nearest_neighbor'],
                marker=titles_2_markers['nearest_neighbor'],
                linewidth=2.0,
                # markevery=(30, 100)
                markevery=(0.05, 0.1)
                )

        # plot the learned variants
        acc = cfg.accuracies[i]
        curr_pac_bayes_results = guarantee_results[i]
        for j in range(len(curr_pac_bayes_results)):
            if cfg.worst_case:
                curr_curve = np.zeros(steps.size)
                curr_size = curr_pac_bayes_results[j].size

                # prob bounds
                curr_curve[:curr_size] = curr_pac_bayes_results[j] #[:cfg.eval_iters]

                # worst-case bounds
                indices = 1 / np.sqrt(steps + 2) * z_star_max * 1.1 < acc
                # cold_start_results[i][:cfg.eval_iters]

                curr_curve[curr_size:] = curr_pac_bayes_results[j].max()

                curr_curve[indices] = 1.0
            else:
                curr_curve = curr_pac_bayes_results[j]
            plt.plot(steps,
                curr_curve, 
                        color=colors[j], 
                        alpha=0.6,
                        # markevery=(0, 100),
                        markevery=0.1,
                        marker=markers[j])
        plt.tight_layout()
        plt.xlabel('evaluation steps')
        plt.ylabel(f"frac. at {acc} fp res")
        plt.xscale('log')
        plt.savefig(f"acc_{acc}.pdf", bbox_inches='tight')
        plt.clf()


def get_worst_case_datetime(example, cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    datetime = cfg.worst_case_datetime
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/z_star_max.csv"

    # fp_file = f"tol={acc}_train.csv" if train else f"tol={acc}_test.csv"
    # df = read_csv(f"{path}/{fp_file}")
    # filename = cfg.worst_case_datetime

    # Open the file in read mode
    with open(path, 'r') as file:
        # reader = csv.reader(file)
        reader = read_csv(file)
        z_star_max = float(reader.columns[0])
        theta_max = reader[str(z_star_max)][0]
        # import pdb
        # pdb.set_trace()
        
        # Read the first row and extract the scalar value
        # for row in reader:
        #     z_star_max = row[0]
    return z_star_max, theta_max

def create_gen_l2o_results(example, cfg):
    # example = 'sparse_coding'
    # overlay_training_losses(example, cfg)
    # create_journal_results(example, cfg, train=False)
    # create_genL2O_results()
    metrics, timing_data, titles = get_all_data(example, cfg, train=False)

    if len(titles) == 4:
        titles[-2] = titles[-2] + '_deterministic'
    nmse = metrics[0]
    for i in range(len(nmse)):
        # if titles[i] != 'cold_start' and titles[i] != 'nearest_neighbor':
        #     plt.plot(nmse[i])
        plt.plot(nmse[i][:cfg.eval_iters],
                 linestyle=titles_2_styles[titles[i]], 
                 color=titles_2_colors[titles[i]],
                 marker=titles_2_markers[titles[i]],
                 markevery=(0, 100)
                 )
    plt.tight_layout()
    plt.xlabel('evaluation steps')
    plt.ylabel("fixed-point residual")
    plt.yscale('log')
    plt.savefig('fp_res.pdf', bbox_inches='tight')
    plt.clf()

    z_star_max, theta_max = get_worst_case_datetime(example, cfg)

    out = get_frac_solved_data(example, cfg)
    all_test_results, all_pac_bayes_results, cold_start_results, nearest_neighbor_results = out
    markers = ['o', 's']
    cmap = plt.cm.Set1
    colors = cmap.colors
    styles = ['-', '-']
    for i in range(len(cfg.accuracies)):
        # plot ista and fista
        mark_start = titles_2_marker_starts['cold_start']
        plt.plot(cold_start_results[i][:cfg.eval_iters], 
                 linestyle=titles_2_styles['cold_start'], 
                 color=titles_2_colors['cold_start'],
                 marker=titles_2_markers['cold_start'],
                #  markevery=(30, 100)
                markevery=0.1
                 )
        mark_start = titles_2_marker_starts['nearest_neighbor']
        plt.plot(nearest_neighbor_results[i][:cfg.eval_iters], 
                linestyle=titles_2_styles['nearest_neighbor'], 
                color=titles_2_colors['nearest_neighbor'],
                marker=titles_2_markers['nearest_neighbor'],
                # markevery=(60, 100)
                markevery=0.1
                )

        # plot the learned variants
        acc = cfg.accuracies[i]
        curr_test_results = all_test_results[i]
        curr_pac_bayes_results = all_pac_bayes_results[i]
        for j in range(len(curr_test_results)):
            curr_size = curr_pac_bayes_results[j].size
            curr_test = np.ones(cfg.eval_iters)
            curr_test[:curr_size] = curr_test_results[j]
            plt.plot(curr_test, 
                     linestyle='-', 
                     color=colors[0], 
                     marker=markers[0],
                     markevery=0.1
                     )
            # curr_pac = curr_pac_bayes_results[j]

            # worst-case
            # prob bounds
            # curr_curve[:curr_size] = curr_pac_bayes_results[j] #[:cfg.eval_iters]

            # worst-case bounds
            
            steps = np.arange(cfg.eval_iters)

            init_diff = z_star_max * 1.1 + 1.1 * theta_max * 30
            indices = .995 ** steps * init_diff  < acc
            # import pdb
            # pdb.set_trace()
            # cold_start_results[i][:cfg.eval_iters]

            curr_pac = np.zeros(cfg.eval_iters)
            curr_pac[:curr_size] = curr_pac_bayes_results[j]
            curr_pac[curr_size:] = curr_pac_bayes_results[j].max()
            curr_pac[indices] = 1.0


            plt.plot(curr_pac, 
                     linestyle='-', 
                     color=colors[1], 
                     markevery=0.1,
                     marker=markers[1])
        plt.tight_layout()
        plt.xlabel('evaluation steps')
        plt.xscale('log')
        plt.ylabel(f"frac. at {acc} fp res")
        plt.savefig(f"acc_{acc}.pdf", bbox_inches='tight')
        plt.clf()
    import pdb
    pdb.set_trace()


def create_gen_l2o_results_maml(example, cfg):
    # example = 'sparse_coding'
    # overlay_training_losses(example, cfg)
    # create_journal_results(example, cfg, train=False)
    # create_genL2O_results()
    metrics, timing_data, titles = get_all_data(example, cfg, train=False)

    if len(titles) == 4:
        titles[-2] = titles[-2] + '_deterministic'
    nmse = metrics[0]
    for i in range(len(nmse)):
        # if titles[i] != 'cold_start' and titles[i] != 'nearest_neighbor':
        #     plt.plot(nmse[i])
        plt.plot(nmse[i][:cfg.eval_iters],
                 linestyle=titles_2_styles[titles[i]], 
                 color=titles_2_colors[titles[i]],
                 marker=titles_2_markers[titles[i]],
                 markevery=(0, 100)
                 )
    plt.tight_layout()
    plt.xlabel('evaluation steps')
    plt.ylabel("fixed-point residual")
    plt.yscale('log')
    plt.savefig('fp_res.pdf', bbox_inches='tight')
    plt.clf()



    out = get_frac_solved_data(example, cfg)
    all_test_results, all_pac_bayes_results, cold_start_results, pretrain_results = out
    markers = ['o', 's']
    cmap = plt.cm.Set1
    colors = cmap.colors
    styles = ['-', '-']
    for i in range(len(cfg.accuracies)):
        # plot ista and fista
        # mark_start = titles_2_marker_starts['cold_start']
        # plt.plot(cold_start_results[i], #[:cfg.eval_iters], 
        #          linestyle=titles_2_styles['cold_start'], 
        #          color=titles_2_colors['cold_start'],
        #          marker=titles_2_markers['cold_start'],
        #          markevery=2
        #          )

        # plot the pretrained model
        plt.plot(pretrain_results[i],
                 linestyle=titles_2_styles['cold_start'], 
                 color=titles_2_colors['cold_start'],
                 marker=titles_2_markers['cold_start'],
                 markevery=2
                 )


        # plot the learned variants
        acc = cfg.accuracies[i]
        curr_test_results = all_test_results[i]
        curr_pac_bayes_results = all_pac_bayes_results[i]
        for j in range(len(curr_test_results)):
            plt.plot(curr_pac_bayes_results[j], 
                     linestyle='-', 
                     color=colors[1], 
                     marker=markers[1],
                     markevery=2
                     )
            plt.plot(curr_test_results[j], 
                     linestyle='-', 
                     color=colors[0], 
                     markevery=2,
                     marker=markers[0])
            
        plt.tight_layout()
        plt.title(r'${}$ loss threshold'.format(acc))
        plt.hlines(0.0, 0, curr_test_results[j].size-1, color='black', linestyle='--', alpha=0.3)
        plt.hlines(1.0, 0, curr_test_results[j].size-1, color='black', linestyle='--', alpha=0.3)
        plt.ylim((-.05, 1.05))
        plt.xlabel('gradient steps')
        plt.ylabel(f"fraction less than {acc} loss")
        plt.savefig(f"acc_{acc}.pdf", bbox_inches='tight')
        plt.clf()
        # import pdb
        # pdb.set_trace()




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
    # plot_eval_iters(example, cfg, train=False)
    overlay_training_losses(example, cfg)
    create_journal_results(example, cfg, train=False)


@hydra.main(config_path='configs/robust_ls', config_name='robust_ls_plot.yaml')
def robust_ls_plot_eval_iters(cfg):
    example = 'robust_ls'
    # overlay_training_losses(example, cfg)
    # # plot_eval_iters(example, cfg, train=False)
    # create_journal_results(example, cfg, train=False)
    create_gen_l2o_results(example, cfg)


@hydra.main(config_path='configs/sparse_pca', config_name='sparse_pca_plot.yaml')
def sparse_pca_plot_eval_iters(cfg):
    example = 'sparse_pca'
    overlay_training_losses(example, cfg)
    # plot_eval_iters(example, cfg, train=False)
    create_journal_results(example, cfg, train=False)


@hydra.main(config_path='configs/lasso', config_name='lasso_plot.yaml')
def lasso_plot_eval_iters(cfg):
    example = 'lasso'
    # overlay_training_losses(example, cfg)
    # # plot_eval_iters(example, cfg, train=False)
    # create_journal_results(example, cfg, train=False)

    # create_gen_l2o_results(example, cfg)
    create_classical_results(example, cfg)


@hydra.main(config_path='configs/unconstrained_qp', config_name='unconstrained_qp_plot.yaml')
def unconstrained_qp_plot_eval_iters(cfg):
    example = 'unconstrained_qp'
    
    # plot_eval_iters(example, cfg, train=False)
    # create_journal_results(example, cfg, train=False)
    # overlay_training_losses(example, cfg)
    create_gen_l2o_results(example, cfg)


@hydra.main(config_path='configs/sine', config_name='sine_plot.yaml')
def sine_plot_eval_iters(cfg):
    example = 'sine'
    # plot_eval_iters(example, cfg, train=False)
    # create_journal_results(example, cfg, train=False)
    # overlay_training_losses(example, cfg)
    create_gen_l2o_results_maml(example, cfg)


@hydra.main(config_path='configs/mpc', config_name='mpc_plot.yaml')
def mpc_plot_eval_iters(cfg):
    example = 'mpc'
    overlay_training_losses(example, cfg)
    # plot_eval_iters(example, cfg, train=False)
    create_journal_results(example, cfg, train=False)
    
@hydra.main(config_path='configs/phase_retrieval', config_name='phase_retrieval_plot.yaml')
def phase_retrieval_plot_eval_iters(cfg):
    example = 'phase_retrieval'
    # plot_eval_iters(example, cfg, train=False)
    
    # plot_eval_iters(example, cfg, train=False)
    create_journal_results(example, cfg, train=False)
    overlay_training_losses(example, cfg)


@hydra.main(config_path='configs/mnist', config_name='mnist_plot.yaml')
def mnist_plot_eval_iters(cfg):
    example = 'mnist'
    # plot_eval_iters(example, cfg, train=False)
    
    # plot_eval_iters(example, cfg, train=False)
    # create_journal_results(example, cfg, train=False)
    # overlay_training_losses(example, cfg)
    create_classical_results(example, cfg)


@hydra.main(config_path='configs/jamming', config_name='jamming_plot.yaml')
def jamming_plot_eval_iters(cfg):
    example = 'jamming'
    # plot_eval_iters(example, cfg, train=False)
    overlay_training_losses(example, cfg)
    # plot_eval_iters(example, cfg, train=False)
    create_journal_results(example, cfg, train=False)


@hydra.main(config_path='configs/quadcopter', config_name='quadcopter_plot.yaml')
def quadcopter_plot_eval_iters(cfg):
    example = 'quadcopter'
    # plot_eval_iters(example, cfg, train=False)
    overlay_training_losses(example, cfg)
    # plot_eval_iters(example, cfg, train=False)
    create_journal_results(example, cfg, train=False)



def plot_sparse_coding(metrics, titles, eval_iters, vert_lines=False):
    """
    metrics is a list of lists

    e.g.
    metrics = [metric_fp, metric_pr, metric_dr]
    metric_fp = [cs, nn-ws, ps-ws, k=5, k=10, ..., k=120]
        where cs is a numpy array
    same for metric_pr and metric_dr

    each metric has a title

    each line within each metric has a style

    note that we do not explicitly care about the k values
        we will manually create the legend in latex later
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12), sharey='row')
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 13), sharey='row')
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12), sharey='row')

    # for i in range(2):

    # yscale
    axes[0, 0].set_yscale('log')
    axes[0, 1].set_yscale('log')

    # x-label
    # axes[0, 0].set_xlabel('evaluation iterations')
    # axes[0, 1].set_xlabel('evaluation iterations')
    fontsize = 40
    title_fontsize = 40
    axes[1, 0].set_xlabel('evaluation iterations', fontsize=fontsize)
    axes[1, 1].set_xlabel('evaluation iterations', fontsize=fontsize)

    # y-label
    axes[0, 0].set_ylabel('test fixed-point residual', fontsize=fontsize)
    axes[1, 0].set_ylabel('test gain to cold start', fontsize=fontsize)

    axes[0, 0].set_title('training with fixed-point residual losses', fontsize=title_fontsize)
    axes[0, 1].set_title('training with regression losses', fontsize=title_fontsize)


    axes[0, 0].set_xticklabels([])
    axes[0, 1].set_xticklabels([])


    if len(metrics) == 3:
        start = 1
    else:
        start = 0

    # plot the fixed-point residual
    for i in range(1):
        curr_metric = metrics[i]
        for j in range(len(curr_metric)):
            title = titles[j]
            color = titles_2_colors[title]
            style = titles_2_styles[title]
            marker = titles_2_markers[title]
            mark_start = titles_2_marker_starts[title]
            if title[:3] != 'reg':
                axes[0, 0].plot(np.array(curr_metric[j])[start:eval_iters + start], 
                                linestyle=style, marker=marker, color=color, 
                                markevery=(2 * mark_start, 2 * 25))
                # if vert_lines:
                #     if title[0] == 'k':
                #         k = int(title[1:])
                #         axes[i].axvline(k, color=color)
            if title[:3] != 'obj':
                axes[0, 1].plot(np.array(curr_metric[j])[start:eval_iters + start], 
                                linestyle=style, marker=marker, color=color, 
                                markevery=(2 * mark_start, 2 * 25))
                # if vert_lines:
                #     if title[0] == 'k':
                #         k = int(title[1:])
                #         axes[i].axvline(k, color=color)

    # plot the gain
    for i in range(1):
        curr_metric = metrics[i]
        for j in range(len(curr_metric)):
            title = titles[j]
            color = titles_2_colors[title]
            style = titles_2_styles[title]
            marker = titles_2_markers[title]
            mark_start = titles_2_marker_starts[title]
            # if j > 0:
            #     gain = cs / np.array(curr_metric[j])[start:eval_iters + start]
            # else:
            #     cs = np.array(curr_metric[j])[start:eval_iters + start]
            if j == 0:
                cs = np.array(curr_metric[j])[start:eval_iters + start]
            else:
                gain = np.clip(cs / np.array(curr_metric[j])[start:eval_iters + start], 
                               a_min=0, a_max=1500)
                if title[:3] != 'reg':
                    axes[1, 0].plot(gain, linestyle=style, marker=marker, color=color, 
                                    markevery=(2 * mark_start, 2 * 25))
                if title[:3] != 'obj':
                    axes[1, 1].plot(gain, linestyle=style, marker=marker, color=color, 
                                    markevery=(2 * mark_start, 2 * 25))

            # if vert_lines:
            #     if title[0] == 'k':
            #         k = int(title[1:])
            #         plt.axvline(k, color=color)
    # plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    
    fig.tight_layout()
    if vert_lines:
        plt.savefig('all_metric_plots_vert.pdf', bbox_inches='tight')
    else:
        plt.savefig('all_metric_plots.pdf', bbox_inches='tight')
    
    plt.clf()



@hydra.main(config_path='configs/all', config_name='plot.yaml')
def plot_l4dc(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
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
        # train_yaml_filename = f"{orig_cwd}/outputs/{example}/train_outputs/
        #                         {datetime}/.hydra/config.yaml"
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
    axes[0].set_ylabel('test fixed-point residuals')
    axes[0].set_title('oscillating masses')
    axes[1].set_title('vehicle')
    axes[2].set_title('markowitz')

    plt.savefig('combined_plots.pdf', bbox_inches='tight')
    fig.tight_layout()




def create_timing_table(timing_data, titles, rel_tols, abs_tols):
    df = pd.DataFrame()

    df['rel_tols'] = rel_tols
    df['abs_tols'] = abs_tols

    for i in range(len(titles)):
        df[titles[i]] = np.round(timing_data[i], decimals=2)

    # for i in range(len(titles)):
    #     df_acc = update_acc(df_acc, accs, titles[i], metrics_fp[i])
    df.to_csv('timings.csv')
    # pdb.set_trace()



# def get_styles_from_titles(titles):
#     style = []
#     return styles


def create_journal_results(example, cfg, train=False):
    """
    does the following steps

    1. get data 
        1.1 (fixed-point residuals, primal residuals, dual residuals) or 
            (fixed-point residuals, obj_diffs)
        store this in metrics

        1.2 timing data
        store this in time_results

        also need: styles, titles
            styles comes from titles
    2. plot the metrics
    3. create the table for fixed-point residuals
    4. create the table for timing results
    """

    # step 1
    metrics, timing_data, titles = get_all_data(example, cfg, train=train)

    # step 2
    plot_all_metrics(metrics, titles, cfg.eval_iters, vert_lines=True)
    plot_all_metrics(metrics, titles, cfg.eval_iters, vert_lines=False)

    # step 3
    metrics_fp = metrics[0]
    create_fixed_point_residual_table(metrics_fp, titles, cfg.accuracies)

    # step 3
    if len(metrics) == 3:
        create_timing_table(timing_data, titles, cfg.rel_tols, cfg.abs_tols)


def determine_scs_or_osqp(example):
    if example == 'unconstrained_qp' or example == 'lasso' or example == 'jamming' or example == 'sparse_coding' or example == 'sine':
        return False
    return True

def get_conv_rates_data(example, cfg):
    orig_cwd = hydra.utils.get_original_cwd()

    cold_start_datetime = cfg.cold_start_datetime
    

    nn_datetime = cfg.nearest_neighbor_datetime
    

    # get the datetimes
    learn_datetimes = cfg.output_datetimes
    if learn_datetimes == []:
        dt = recover_last_datetime(orig_cwd, example, 'train')
        learn_datetimes = [dt]

    # all_test_results = []
    # all_pac_bayes_results = []
    # cold_start_results = []
    # nearest_neighbor_results = []

    # for acc in cfg.accuracies:
    if cold_start_datetime != '':
        curr_cold_start_results, conv_rates = load_conv_rates(example, cold_start_datetime, 
                                                              train=False, title='no_train')
        # cold_start_results.append(curr_cold_start_results)
    if nn_datetime != '':
        curr_nearest_neighbor_results, _ = load_conv_rates(example, nn_datetime, 
                                                           train=False, title='nearest_neighbor')
        # nearest_neighbor_results.append(curr_nearest_neighbor_results)
    curr_pac_bayes_results = []
    curr_test_results = []
    for datetime in learn_datetimes:
        pac_bayes_curve, _ = load_conv_rates(example, datetime, train=True)
        test_curve, _ = load_conv_rates(example, datetime, train=False)
        curr_pac_bayes_results.append(pac_bayes_curve)
        curr_test_results.append(test_curve)
    # all_pac_bayes_results.append(curr_pac_bayes_results)
    # all_test_results.append(curr_test_results)
    return conv_rates, curr_test_results, curr_pac_bayes_results, curr_cold_start_results, curr_nearest_neighbor_results


def get_frac_solved_data_classical(example, cfg):
    # setup
    orig_cwd = hydra.utils.get_original_cwd()

    cold_start_datetimes = cfg.cold_start_datetimes
    
    

    # get the datetimes
    # learn_datetimes = cfg.output_datetimes
    # if learn_datetimes == []:
    #     dt = recover_last_datetime(orig_cwd, example, 'train')
    #     learn_datetimes = [dt]

    # all_test_results = []
    # all_pac_bayes_results = []
    cold_start_results = []
    guarantee_results = []
    # nearest_neighbor_results = []
    for acc in cfg.accuracies:
    # for i in range(len(cfg.accuracies)):
    #     acc = cfg.accuracies[i]
        # if cold_start_datetime != '':
        # cold_start_datetime = recover_last_datetime(orig_cwd, example, 'train')
        curr_cold_start_results = load_frac_solved(example, cold_start_datetimes[0], acc, train=True, 
                                                    title='no_train')
        cold_start_results.append(curr_cold_start_results)
        curr_guarantee_results = []
        for datetime in cold_start_datetimes:
            single_guarantee_results = load_frac_solved(example, datetime, acc, train=True, 
                                                        title='no_train_pac_bayes')
            curr_guarantee_results.append(single_guarantee_results)
        guarantee_results.append(curr_guarantee_results)
        # if nn_datetime != '':
        #     # nn_datetime = recover_last_datetime(orig_cwd, example, 'train')
        #     curr_nearest_neighbor_results = load_frac_solved(example, nn_datetime, acc, train=False, title='nearest_neighbor')
        #     nearest_neighbor_results.append(curr_nearest_neighbor_results)
        # curr_pac_bayes_results = []
        # curr_test_results = []
        # for datetime in learn_datetimes:
        #     pac_bayes_curve = load_frac_solved(example, datetime, acc, train=True)
        #     test_curve = load_frac_solved(example, datetime, acc, train=False)
        #     curr_pac_bayes_results.append(pac_bayes_curve)
        #     curr_test_results.append(test_curve)
        # all_pac_bayes_results.append(curr_pac_bayes_results)
        # all_test_results.append(curr_test_results)
    return cold_start_results, guarantee_results


def get_frac_solved_data(example, cfg):
    # setup
    orig_cwd = hydra.utils.get_original_cwd()

    cold_start_datetime = cfg.cold_start_datetime
    

    nn_datetime = cfg.nearest_neighbor_datetime

    pretrain_datetime = cfg.pretrain_datetime

    

    # get the datetimes
    learn_datetimes = cfg.output_datetimes
    if learn_datetimes == []:
        dt = recover_last_datetime(orig_cwd, example, 'train')
        learn_datetimes = [dt]

    all_test_results = []
    all_pac_bayes_results = []
    cold_start_results = []
    nearest_neighbor_results = []
    for acc in cfg.accuracies:
        if cold_start_datetime != '':
            # cold_start_datetime = recover_last_datetime(orig_cwd, example, 'train')
            curr_cold_start_results = load_frac_solved(example, cold_start_datetime, acc, train=False, title='no_train')
            cold_start_results.append(curr_cold_start_results)
        if nn_datetime != '':
            # nn_datetime = recover_last_datetime(orig_cwd, example, 'train')
            curr_nearest_neighbor_results = load_frac_solved(example, nn_datetime, acc, train=False, title='nearest_neighbor')
            nearest_neighbor_results.append(curr_nearest_neighbor_results)
        if pretrain_datetime != '':
            # nn_datetime = recover_last_datetime(orig_cwd, example, 'train')
            # curr_nearest_neighbor_results = load_frac_solved(example, nn_datetime, acc, train=False, title='nearest_neighbor')
            curr_nearest_neighbor_results = load_frac_solved(example, pretrain_datetime, acc, train=False)
            nearest_neighbor_results.append(curr_nearest_neighbor_results)
        curr_pac_bayes_results = []
        curr_test_results = []
        for datetime in learn_datetimes:
            pac_bayes_curve = load_frac_solved(example, datetime, acc, train=True)
            test_curve = load_frac_solved(example, datetime, acc, train=False)
            curr_pac_bayes_results.append(pac_bayes_curve)
            curr_test_results.append(test_curve)
        all_pac_bayes_results.append(curr_pac_bayes_results)
        all_test_results.append(curr_test_results)
    return all_test_results, all_pac_bayes_results, cold_start_results, nearest_neighbor_results

def get_all_data(example, cfg, train=False):
    # setup
    orig_cwd = hydra.utils.get_original_cwd()

    # get the datetimes
    learn_datetimes = cfg.output_datetimes
    if learn_datetimes == []:
        dt = recover_last_datetime(orig_cwd, example, 'train')
        learn_datetimes = [dt]

    cold_start_datetime = cfg.cold_start_datetime
    if cold_start_datetime == '':
        cold_start_datetime = recover_last_datetime(orig_cwd, example, 'train')

    nn_datetime = cfg.nearest_neighbor_datetime
    if nn_datetime == '':
        nn_datetime = recover_last_datetime(orig_cwd, example, 'train')

    metrics_list = []
    timing_data = []

    # check if prev_sol exists
    # if 'prev_sol' in cfg.keys():
    # prev_sol_bool = cfg.prev_sol_datetime
    prev_sol_bool = 'prev_sol_datetime' in cfg.keys()

    # benchmarks = ['cold_start', 'nearest_neighbor']
    # benchmark_dts = [cold_start_datetime, nn_datetime]
    benchmarks, benchmark_dts = [], []
    if 'cold_start_datetime' in cfg.keys():
        benchmarks.append('cold_start')
        benchmark_dts.append(cold_start_datetime)
    if 'nearest_neighbor_datetime' in cfg.keys() and example != 'sine':
        benchmarks.append('nearest_neighbor')
        benchmark_dts.append(nn_datetime)
    if prev_sol_bool:
        benchmarks.append('prev_sol')
        benchmark_dts.append(cfg.prev_sol_datetime)

    # for init in ['cold_start', 'nearest_neighbor', 'prev_sol']:
    for i in range(len(benchmarks)):
        init = benchmarks[i]
        datetime = benchmark_dts[i]
        metric, timings = load_data_per_title(example, init, datetime)
        metrics_list.append(metric)
        timing_data.append(timings)

    # learned warm-starts
    k_vals = np.zeros(len(learn_datetimes))
    loss_types = []
    for i in range(len(k_vals)):
        datetime = learn_datetimes[i]
        loss_type = get_loss_type(orig_cwd, example, datetime)
        loss_types.append(loss_type)
        k = get_k(orig_cwd, example, datetime)
        k_vals[i] = k
        metric, timings = load_data_per_title(example, k, datetime)
        metrics_list.append(metric)
        timing_data.append(timings)

    k_vals_new = []
    for i in range(k_vals.size):
        k = k_vals[i]
        new_k = k if k >= 2 else 0
        k_vals_new.append(new_k)
    # titles = benchmarks + [f"k{int(k)}" for k in k_vals_new]
    titles = benchmarks
    for i in range(len(loss_types)):
        loss_type = loss_types[i]
        k = k_vals_new[i]
        titles.append(f"{loss_type}_k{int(k)}")

    # titles = cfg.loss_overlay_titles
    # for i in range(len(datetimes)):
    #     datetime = datetimes[i]
    #     path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/{iters_file}"
    #     df = read_csv(path)

    #     '''
    #     for the fully trained models, track the k value
    #     - to do this, load the train_yaml file
    #     '''
    #     train_yaml_filename = f"{orig_cwd}/outputs/{example}/train_outputs/
    #                           {datetime}/.hydra/config.yaml"
    #     with open(train_yaml_filename, "r") as stream:
    #         try:
    #             out_dict = yaml.safe_load(stream)
    #         except yaml.YAMLError as exc:
    #             print(exc)
    #     k = out_dict['train_unrolls']
    #     k_vals[i] = k

    #     last_column = df.iloc[:, -1]

    # combine metrics
    # metrics = combine_metrics(metrics_list)
    metrics = [[row[i] for row in metrics_list] for i in range(len(metrics_list[0]))]

    return metrics, timing_data, titles


# def combine_metrics(metrics_list):
#     """
#     metrics_list = [cs_metric, nn_metric, k10_metric]

#     metrics = [fp, pr, dr]
#     """
#     metrics = []
#     for i in range(len(metrics_list)):
#         for j in range(len(metrics_list[i])):
#             metrics[i]
#     return metrics

def load_conv_rates(example, datetime, train, title=None):
    orig_cwd = hydra.utils.get_original_cwd()
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}"

    fp_file = "conv_rates_train.csv" if train else "conv_rates_test.csv"
    df = read_csv(f"{path}/{fp_file}")
    if title is None:
        if train:
            results = df.iloc[:, -1]
        else:
            results = df.iloc[:, -2]
    else:
        results = df[title]
    conv_rates = df.iloc[:, 1]

    return results, conv_rates


def load_frac_solved(example, datetime, acc, train, title=None):
    orig_cwd = hydra.utils.get_original_cwd()
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/frac_solved"

    fp_file = f"tol={acc}_train.csv" if train else f"tol={acc}_test.csv"
    df = read_csv(f"{path}/{fp_file}")
    if title is None:
        if train:
            results = df.iloc[:, -1]
        else:
            results = df.iloc[:, -3]
    else:
        results = df[title]
    return results

def load_data_per_title(example, title, datetime, train=False):
    scs_or_osqp = determine_scs_or_osqp(example)

    orig_cwd = hydra.utils.get_original_cwd()
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}"
    # no_learning_path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/{iters_file}"

    # read the eval iters csv
    # fp_file = 'eval_iters_train.csv' if train else 'eval_iters_test.csv'
    fp_file = 'iters_compared_train.csv' if train else 'iters_compared_test.csv'
    fp_df = read_csv(f"{path}/{fp_file}")
    fp = get_eval_array(fp_df, title)
    # fp = fp_df[title]

    # read the primal and dual residausl csv
    if scs_or_osqp:
        pr_file = 'primal_residuals_train.csv' if train else 'primal_residuals_test.csv'
        pr_df = read_csv(f"{path}/{pr_file}")
        pr = get_eval_array(pr_df, title)
        # pr = pr_df[title]

        dr_file = 'dual_residuals_train.csv' if train else 'dual_residuals_test.csv'
        dr_df = read_csv(f"{path}/{dr_file}")
        # dr = dr_df[title]
        dr = get_eval_array(dr_df, title)
        metric = [fp, pr, dr]

    # read the obj_diffs csv
    else:
        # obj_file = 'obj_vals_diff_train.csv' if train else 'obj_vals_diff_test.csv'
        # obj_df = read_csv(f"{path}/{obj_file}")
        # # obj = obj_df[title]
        # obj = get_eval_array(obj_df, title)
        # metric = [fp, obj]
        metric = [fp, fp]
    
    # do timings
    try:
        if scs_or_osqp:
            train_str = 'train' if train else 'test'
            timings_file = f"solve_C/{train_str}_aggregate_solve_times.csv"
            timings_df = read_csv(f"{path}/{timings_file}")
            # timings = timings_df[title]
            timings = get_eval_array(timings_df, title)
        else:
            timings = None
    except:
        timings = None

    return metric, timings





def get_eval_array(df, title):
    if title == 'cold_start' or title == 'no_learn':
        data = df['no_train']
    elif title == 'nearest_neighbor':
        data = df['nearest_neighbor']
    elif title == 'prev_sol':
        data = df['prev_sol']
    else:
        # case of the learned warm-start, take the latest column
        data = df.iloc[:, -1]
    return data


# def load_learned_data(datetime):
#     return metric, timings


def create_fixed_point_residual_table(metrics_fp, titles, accs):
    # create pandas dataframe
    df_acc = pd.DataFrame()
    df_percent = pd.DataFrame()
    df_acc_both = pd.DataFrame()

    # df_acc
    df_acc['accuracies'] = np.array(accs)
    for i in range(len(titles)):
        df_acc = update_acc(df_acc, accs, titles[i], metrics_fp[i])
    df_acc.to_csv('accuracies.csv')

    # df_percent
    df_percent['accuracies'] = np.array(accs)
    no_learning_acc = df_acc['cold_start']
    for col in df_acc.columns:
        if col != 'accuracies':
            val = 1 - df_acc[col] / no_learning_acc
            df_percent[col] = np.round(val, decimals=2)
    df_percent.to_csv('iteration_reduction.csv')

    # save both iterations and fraction reduction in single table
    # df_acc_both['accuracies'] = df_acc['cold_start']
    # df_acc_both['cold_start_iters'] = np.array(accs)
    df_acc_both['accuracies'] = np.array(accs)
    df_acc_both['cold_start_iters'] = df_acc['cold_start']

    for col in df_percent.columns:
        if col != 'accuracies' and col != 'cold_start':
            df_acc_both[col + '_iters'] = df_acc[col]
            df_acc_both[col + '_red'] = df_percent[col]
    df_acc_both.to_csv('accuracies_reduction_both.csv')



def plot_all_metrics(metrics, titles, eval_iters, vert_lines=False):
    """
    metrics is a list of lists

    e.g.
    metrics = [metric_fp, metric_pr, metric_dr]
    metric_fp = [cs, nn-ws, ps-ws, k=5, k=10, ..., k=120]
        where cs is a numpy array
    same for metric_pr and metric_dr

    each metric has a title

    each line within each metric has a style

    note that we do not explicitly care about the k values
        we will manually create the legend in latex later
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12), sharey='row')
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 13), sharey='row')
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12), sharey='row')

    # for i in range(2):

    # yscale
    axes[0, 0].set_yscale('log')
    axes[0, 1].set_yscale('log')

    # x-label
    # axes[0, 0].set_xlabel('evaluation iterations')
    # axes[0, 1].set_xlabel('evaluation iterations')
    fontsize = 40
    title_fontsize = 40
    axes[1, 0].set_xlabel('evaluation iterations', fontsize=fontsize)
    axes[1, 1].set_xlabel('evaluation iterations', fontsize=fontsize)

    # y-label
    # axes[0, 0].set_ylabel('fixed-point residual')
    # axes[1, 0].set_ylabel('gain to cold start')
    axes[0, 0].set_ylabel('test fixed-point residual', fontsize=fontsize)
    axes[1, 0].set_ylabel('test gain to cold start', fontsize=fontsize)

    # axes[0, 0].set_title('fixed-point residual losses')
    # axes[0, 1].set_title('regression losses')
    # axes[1, 0].set_title('fixed-point residual losses')
    # axes[1, 1].set_title('regression losses')
    axes[0, 0].set_title('training with fixed-point residual losses', fontsize=title_fontsize)
    axes[0, 1].set_title('training with regression losses', fontsize=title_fontsize)
    # axes[1, 0].set_title('training with fixed-point residual losses')
    # axes[1, 1].set_title('training with regression losses')

    axes[0, 0].set_xticklabels([])
    axes[0, 1].set_xticklabels([])

    # axes[0, 0].tick_params(axis='y', which='major', pad=15)
    # axes[1, 0].tick_params(axis='y', which='major', pad=15)

    # titles
    # axes[0, 0].set_title('fixed-point residuals with fixed-point residual-based losses')
    # axes[0, 1].set_title('fixed-point residuals with regression-based losses')
    # axes[1, 0].set_title('gain to cold start with fixed-point residual-based losses')
    # axes[1, 1].set_title('gain to cold start with regression-based losses')

    if len(metrics) == 3:
        start = 1
    else:
        start = 0

    # plot the fixed-point residual
    for i in range(1):
        curr_metric = metrics[i]
        for j in range(len(curr_metric)):
            title = titles[j]
            color = titles_2_colors[title]
            style = titles_2_styles[title]
            marker = titles_2_markers[title]
            mark_start = titles_2_marker_starts[title]
            if title[:3] != 'reg':
                axes[0, 0].plot(np.array(curr_metric[j])[start:eval_iters + start], 
                                linestyle=style, marker=marker, color=color, 
                                markevery=(2 * mark_start, 2 * 25))
                # if vert_lines:
                #     if title[0] == 'k':
                #         k = int(title[1:])
                #         axes[i].axvline(k, color=color)
            if title[:3] != 'obj':
                axes[0, 1].plot(np.array(curr_metric[j])[start:eval_iters + start], 
                                linestyle=style, marker=marker, color=color, 
                                markevery=(2 * mark_start, 2 * 25))
                # if vert_lines:
                #     if title[0] == 'k':
                #         k = int(title[1:])
                #         axes[i].axvline(k, color=color)

    # plot the gain
    for i in range(1):
        curr_metric = metrics[i]
        for j in range(len(curr_metric)):
            title = titles[j]
            color = titles_2_colors[title]
            style = titles_2_styles[title]
            marker = titles_2_markers[title]
            mark_start = titles_2_marker_starts[title]
            # if j > 0:
            #     gain = cs / np.array(curr_metric[j])[start:eval_iters + start]
            # else:
            #     cs = np.array(curr_metric[j])[start:eval_iters + start]
            if j == 0:
                cs = np.array(curr_metric[j])[start:eval_iters + start]
            else:
                gain = np.clip(cs / np.array(curr_metric[j])[start:eval_iters + start], 
                               a_min=0, a_max=1500)
                if title[:3] != 'reg':
                    axes[1, 0].plot(gain, linestyle=style, marker=marker, color=color, 
                                    markevery=(2 * mark_start, 2 * 25))
                if title[:3] != 'obj':
                    axes[1, 1].plot(gain, linestyle=style, marker=marker, color=color, 
                                    markevery=(2 * mark_start, 2 * 25))

            # if vert_lines:
            #     if title[0] == 'k':
            #         k = int(title[1:])
            #         plt.axvline(k, color=color)
    # plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    
    fig.tight_layout()
    if vert_lines:
        plt.savefig('all_metric_plots_vert.pdf', bbox_inches='tight')
    else:
        plt.savefig('all_metric_plots.pdf', bbox_inches='tight')
    
    plt.clf()




    # now plot the gain on a non-log plot
    # plot the gain
    for i in range(1):
        curr_metric = metrics[i]
        for j in range(len(curr_metric)):
        # for j in range(1):
            title = titles[j]
            # title = 'gain to cold start'
            color = titles_2_colors[title]
            style = titles_2_styles[title]

            if j > 0:
                gain = cs / np.array(curr_metric[j])[start:eval_iters + start]
                plt.plot(gain, linestyle=style, color=color)
            else:
                cs = np.array(curr_metric[j])[start:eval_iters + start]
            if vert_lines:
                if title[0] == 'k':
                    k = int(title[1:])
                    # plt.vlines(k, 0, 1000, color=color)
                    plt.axvline(k, color=color)
    plt.ylabel('gain')
    plt.xlabel('evaluation steps')
    if vert_lines:
        plt.savefig('test_gain_plots_vert.pdf', bbox_inches='tight')
    else:
        plt.savefig('test_gain_plots.pdf', bbox_inches='tight')
    fig.tight_layout()


    # plot the loss and the gain for each loss separately
    for i in range(2):
        # fig_width = 9
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 12), sharey='row') #, sharey=True)

        # for i in range(2):

        # yscale
        axes[0].set_yscale('log')
        # axes[0, 1].set_yscale('log')

        # x-label
        # axes[0].set_xlabel('evaluation iterations', fontsize=fontsize)
        # axes[0, 1].set_xlabel('evaluation iterations')
        axes[1].set_xlabel('evaluation iterations', fontsize=fontsize)
        # axes[1, 1].set_xlabel('evaluation iterations')

        # y-label
        axes[0].set_ylabel('fixed-point residual', fontsize=fontsize)
        axes[1].set_ylabel('gain to cold start', fontsize=fontsize)

        axes[0].set_xticklabels([])

        # axes[0, 0].set_title('fixed-point residual losses')
        # axes[0, 1].set_title('regression losses')
        # axes[1, 0].set_title('fixed-point residual losses')
        # axes[1, 1].set_title('regression losses')

        curr_metric = metrics[0]

        for j in range(len(curr_metric)):
            title = titles[j]
            color = titles_2_colors[title]
            style = titles_2_styles[title]
            marker = titles_2_markers[title]
            mark_start = titles_2_marker_starts[title]
            if title[:3] != 'reg' and i == 0:
                # either obj or baselines
                axes[0].plot(np.array(curr_metric[j])[start:eval_iters + start], linestyle=style, marker=marker, color=color, markevery=(2 * mark_start, 2 * 25))
            if title[:3] != 'obj' and  i == 1:
                # either reg or baselines
                axes[0].plot(np.array(curr_metric[j])[start:eval_iters + start], linestyle=style,   marker=marker, color=color, markevery=(2 * mark_start, 2 * 25))

                

        for j in range(len(curr_metric)):
            title = titles[j]
            color = titles_2_colors[title]
            style = titles_2_styles[title]
            marker = titles_2_markers[title]
            mark_start = titles_2_marker_starts[title]
            if j == 0:
                cs = np.array(curr_metric[j])[start:eval_iters + start]
            gain = np.clip(cs / np.array(curr_metric[j])[start:eval_iters + start], 
                           a_min=0, a_max=1500)
            if title[:3] != 'reg' and i == 0:
                axes[1].plot(gain, linestyle=style, marker=marker, color=color, markevery=(2 * mark_start, 2 * 25))
            if title[:3] != 'obj' and i == 1:
                axes[1].plot(gain, linestyle=style, marker=marker, color=color, markevery=(2 * mark_start, 2 * 25))

        if i == 0:
            plt.savefig('fixed_point_residual_loss.pdf', bbox_inches='tight')
        elif i == 1:
            plt.savefig('regression_loss.pdf', bbox_inches='tight')



def get_loss_type(orig_cwd, example, datetime):
    train_yaml_filename = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/.hydra/config.yaml" # noqa
    with open(train_yaml_filename, "r") as stream:
        try:
            out_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # k = int(out_dict['train_unrolls'])
    loss_type = 'reg' if bool(out_dict['supervised']) else 'obj'
    return loss_type



def get_k(orig_cwd, example, datetime):
    train_yaml_filename = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/.hydra/config.yaml" # noqa
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
    # all_train_losses = []
    # all_test_losses = []
    # for i in range(len(datetimes)):
    #     datetime = datetimes[i]
    #     train_losses, test_losses = get_loss_data(example, datetime)
    #     all_train_losses.append(train_losses)
    #     all_test_losses.append(test_losses)

    """"
    now create table like
    k                   5       15      50  
    train reduction
    test reduction
    reduction gap
    """
    # relative_loss_df = pd.DataFrame()
    # relative_loss_df['rows'] = ['relative_train_loss', 'relative_test_loss', 'relative_gap']
    # k_values = []
    # # gain_gaps = []
    # gain_ratios = []
    # # rel_tr_losses = []
    # # rel_te_losses = []
    # for i in range(len(datetimes)):
    #     tr_losses = all_train_losses[i]
    #     te_losses = all_test_losses[i]
    #     orig_loss = te_losses[0]
    #     print(f"tr_losses", tr_losses)
    #     print(f"te_losses", te_losses)
    #     print(f"orig_loss: {orig_loss}")
        
    #     train_gain = orig_loss / tr_losses[-1:].mean()
    #     test_gain = orig_loss / te_losses.iloc[-1]
    #     gain_gap = train_gain - test_gain
    #     gain_ratio = test_gain / train_gain

    #     k = get_k(orig_cwd, example, datetimes[i])
    #     print('k_val', k)
    #     k_values.append(k)
    #     col = f"k = {k}"

    #     row = np.array([train_gain, test_gain, gain_gap])
    #     gain_gaps.append(gain_gap)
    #     gain_ratios.append(gain_ratio)
    #     rel_tr_losses.append(train_gain)
    #     rel_te_losses.append(test_gain)
    #     print(f"row: {row}")
    #     relative_loss_df[col] = np.round(row, decimals=3)
    # relative_loss_df.to_csv('relative_losses.csv')


    gain_ratios = []
    for i in range(len(datetimes)):
        datetime = datetimes[i]
        k = get_k(orig_cwd, example, datetimes[i])
        title = k
        metric_train, timings_train = load_data_per_title(example, title, datetime, train=True)
        fp_res_train = metric_train[0]
        metric_test, timings_test = load_data_per_title(example, title, datetime, train=False)
        fp_res_test = metric_test[0]
        gain_ratio = fp_res_test[k - 1] / fp_res_train[k - 1]
        # gain_ratios[i] = gain_ratio
        gain_ratios.append(gain_ratio)

    k_vals_obj,  gain_ratios_obj, k_vals_reg, gain_ratios_reg = [], [], [], []
    for i in range(len(datetimes)):
        datetime = datetimes[i]
        k = get_k(orig_cwd, example, datetimes[i])
        loss_type = get_loss_type(orig_cwd, example, datetime)
        gain_ratio = gain_ratios[i]
        if loss_type == 'obj':
            k_vals_obj.append(k)
            gain_ratios_obj.append(gain_ratio)
        elif loss_type == 'reg':
            k_vals_reg.append(k)
            gain_ratios_reg.append(gain_ratio)
    create_train_test_plots(example, k_vals_obj,  gain_ratios_obj, k_vals_reg, gain_ratios_reg)

    # # now give a plot for the generalization
    # plt.plot(np.array(k_values), np.array(gain_gaps))
    # plt.xlabel('k')
    # plt.ylabel('gain gap')
    # plt.savefig('gain_gaps.pdf')
    # plt.clf()

    # plt.plot(np.array(k_values), np.array(gain_ratios))
    # plt.xlabel('k')
    # plt.ylabel('gain ratio')
    # plt.savefig('gain_ratios.pdf')
    # plt.clf()

    # # plot the relative train and test final losses
    # # now give a plot for the generalization
    # plt.plot(np.array(k_values), np.array(rel_tr_losses), label='train')
    # plt.plot(np.array(k_values), np.array(rel_te_losses), label='test')
    # plt.xlabel('k')
    # plt.ylabel('relative final losses')
    # plt.legend()
    # plt.savefig('relative_final_losses.pdf')
    # plt.clf()


def create_title(example):
    if example == 'robust_kalman':
        title = 'Robust Kalman filtering'
    elif example == 'robust_ls':
        title = 'Robust non-negative least squares'
    elif example == 'sparse_pca':
        title = 'Sparse PCA'
    elif example == 'phase_retrieval':
        title = 'Phase retrieval'
    elif example == 'mnist':
        title = 'Image deblurring'
    elif example == 'quadcopter':
        title = 'Quadcopter'
    elif example == 'lasso':
        title = 'Lasso'
    elif example == 'unconstrained_wp':
        title = 'Unconstrained QP'
    return title


def create_train_test_plots(example, k_vals_obj,  gain_ratios_obj, k_vals_reg, gain_ratios_reg):
    # now give a plot for the generalization
    plt.figure(figsize=(8, 6))
    plt.plot(np.array(k_vals_obj), np.array(gain_ratios_obj) - 1, label='obj')
    plt.plot(np.array(k_vals_reg), np.array(gain_ratios_reg) - 1, label='reg')
    plt.xlabel(r'$k$')
    plt.ylabel('relative gap')
    # plt.legend()
    title = create_title(example)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('gain_ratios.pdf', bbox_inches='tight')
    plt.clf()

    # plt.plot(np.array(k_values), np.array(gain_ratios))
    # plt.xlabel('k')
    # plt.ylabel('gain ratio')
    # plt.savefig('gain_ratios.pdf')
    # plt.clf()

    # plot the relative train and test final losses
    # now give a plot for the generalization
    # plt.plot(np.array(k_values), np.array(rel_tr_losses), label='train')
    # plt.plot(np.array(k_values), np.array(rel_te_losses), label='test')
    # plt.xlabel('k')
    # plt.ylabel('relative final losses')
    # plt.legend()
    # plt.savefig('relative_final_losses.pdf')
    # plt.clf()



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

    # no learning
    no_learning_path = f"{orig_cwd}/outputs/{example}/train_outputs/{no_learning_datetime}/{iters_file}"  # noqa
    no_learning_df = read_csv(no_learning_path)
    last_column = no_learning_df['no_train']
    plt.plot(last_column[:eval_iters], 'k-.', label='no learning')
    second_derivs_no_learn = second_derivative_fn(np.log(last_column[:eval_iters]))
    df_acc = update_acc(df_acc, accs, 'no_learn', last_column[:eval_iters])

    # naive warm start
    naive_ws_path = f"{orig_cwd}/outputs/{example}/train_outputs/{naive_ws_datetime}/{iters_file}"
    naive_ws_df = read_csv(naive_ws_path)
    # last_column = naive_ws_df['fixed_ws']
    last_column = naive_ws_df['nearest_neighbor']
    # plt.plot(last_column[:eval_iters], 'm-.', label='naive warm start')
    plt.plot(last_column[:eval_iters], 'm-.', label='nearest neighbor')
    second_derivative_fn(np.log(last_column[:eval_iters]))
    df_acc = update_acc(df_acc, accs, 'naive_ws', last_column[:eval_iters])

    # pretraining
    if pretrain_datetime != '':
        pretrain_path = f"{orig_cwd}/outputs/{example}/train_outputs/{pretrain_datetime}/{iters_file}" # noqa
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
        train_yaml_filename = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/.hydra/config.yaml" # noqa
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
    plt.ylabel('test fixed-point residuals')
    plt.legend()
    plt.savefig('eval_iters.pdf', bbox_inches='tight')
    plt.clf()


    # save the iterations required to reach a certain accuracy
    df_acc.to_csv('accuracies.csv')
    df_percent = pd.DataFrame()
    df_percent['accuracies'] = np.array(accs)
    no_learning_acc = df_acc['no_learn']
    for col in df_acc.columns:
        if col != 'accuracies':
            val = 1 - df_acc[col] / no_learning_acc
            df_percent[col] = np.round(val, decimals=2)
    
    df_percent.to_csv('iteration_reduction.csv')

    # save both iterations and fraction reduction in single table
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
    plt.plot(second_derivs_no_learn[5:], label="no learning")
    # if pretrain_datetime != '':
    #     plt.plot(second_derivs_pretrain, label="pretraining")

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
    elif sys.argv[1] == 'sparse_pca':
        sys.argv[1] = base + 'sparse_pca/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        sparse_pca_plot_eval_iters()
    elif sys.argv[1] == 'lasso':
        sys.argv[1] = base + 'lasso/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        lasso_plot_eval_iters()
    elif sys.argv[1] == 'unconstrained_qp':
        sys.argv[1] = base + 'unconstrained_qp/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        unconstrained_qp_plot_eval_iters()
    elif sys.argv[1] == 'mpc':
        sys.argv[1] = base + 'mpc/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        mpc_plot_eval_iters()
    elif sys.argv[1] == 'mnist':
        sys.argv[1] = base + 'mnist/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        mnist_plot_eval_iters()
    elif sys.argv[1] == 'quadcopter':
        sys.argv[1] = base + 'quadcopter/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        quadcopter_plot_eval_iters()
    elif sys.argv[1] == 'sparse_coding':
        sys.argv[1] = base + 'sparse_coding/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        sparse_coding_plot_eval_iters()
    elif sys.argv[1] == 'sine':
        sys.argv[1] = base + 'sine/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        sine_plot_eval_iters()
    elif sys.argv[1] == 'jamming':
        sys.argv[1] = base + 'jamming/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        jamming_plot_eval_iters()
    elif sys.argv[1] == 'all':
        sys.argv[1] = base + 'all/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        plot_l4dc()
