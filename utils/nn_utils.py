import numpy as np
import pandas as pd
import os
import pdb
import matplotlib.pyplot as plt
from jax import random, jit
import jax.numpy as jnp
import functools


def save_nonconvex_result_2_csv(X_stars, opt_vals, times, MIPgaps,
                                problem_type, save_idx, slurm_idx=0, warm=False):
    '''
    X_stars is a numpy array of size (N, n)
    times is a numpy array of size (n)
    opt_vals is a numpy array of size (N)

    this method will save X_stars and times to the csv file

        time   X_star(1) ..... X_star(n)
    1
    2
    3
    ...
    N

    e.g.
    data/maxcut/experiment3/non_convex_solution.csv

    manually write a README.txt file that describes the experiment

    need to
    - count the number of existing files in that location
    '''

    N, n = X_stars.shape
    data = np.hstack((np.reshape(times, (N, 1)),
                     np.reshape(opt_vals, (N, 1)),
                     np.reshape(MIPgaps, (N, 1)),
                     X_stars))
    columns = ['time', 'opt_val', 'mip_gap']
    for i in range(n):
        columns.append('X_star_' + str(i))
    df = pd.DataFrame(data, columns=columns)

    directory = 'data/' + problem_type

    list_ = os.listdir(directory)  # dir is your directory path
    if '.DS_Store' in list_:
        list_.remove('.DS_Store')

    # if warm:
    #     num_experiments = len(list_) - 1
    # else:
    #     num_experiments = len(list_) - 1
        # os.mkdir('data/' + problem_type + '/experiment_' +
        #      str(num_experiments) + '/')

    if slurm_idx is None:
        if warm:
            file_name = 'data/' + problem_type + '/experiment_' + \
                str(save_idx) + '/warm_soln.csv' # str(num_experiments) + '/warm_soln.csv'

        else:
            file_name = 'data/' + problem_type + '/experiment_' + \
                str(save_idx) + '/non_convex_soln.csv'
    else:
        if warm:
            file_name = 'data/' + problem_type + '/experiment_' + \
                str(save_idx) + '/' + str(slurm_idx) + '/warm_soln.csv'
        else:
            file_name = 'data/' + problem_type + '/experiment_' + \
                str(save_idx) + '/' + str(slurm_idx) + '/non_convex_soln.csv'
    df.to_csv(file_name)


def load_nonconvex_result_csv(problem_type, experiment_num, slurm_idx=0, warm=False):
    if slurm_idx is None:
        if warm:
            file_name = 'data/' + problem_type + '/experiment_' + \
                str(experiment_num) + '/warm_soln.csv'
        else:
            file_name = 'data/' + problem_type + '/experiment_' + \
                str(experiment_num) + '/non_convex_soln.csv'
    else:
        if warm:
            file_name = 'data/' + problem_type + '/experiment_' + \
                str(experiment_num) + '/' + str(slurm_idx) + '/warm_soln.csv'
        else:
            file_name = 'data/' + problem_type + '/experiment_' + \
                str(experiment_num) + '/' + str(slurm_idx) + '/non_convex_soln.csv'
    # file_name = 'data/non_convex_soln/' + problem_type + \
    #     '/experiment_' + str(experiment_num) + '.csv'
    df = pd.read_csv(file_name)
    data = df.to_numpy()
    '''
    the zeroth column is just the problem number
    the first column is the time
    the rest is x_star
    '''
    times = data[:, 1]
    opt_vals = data[:, 2]
    MIPgaps = data[:, 3]
    X_stars = data[:, 4:]
    return X_stars, opt_vals, times, MIPgaps




def plot_path_planning(centers, radii, x, squares=True):
    # if square, radii = length
    num_obs = radii.size
    n = x.size
    T = int(n/2)
    fig, ax = plt.subplots()
    for j in range(num_obs):
        if squares:
            ax.add_patch(plt.Rectangle(
                (centers[j, :] - radii[j]/2), radii[j], radii[j], facecolor='none', edgecolor='r'))
        else:
            ax.add_patch(plt.Circle(
                (centers[j, :]), radii[j], facecolor='none', edgecolor='r'))
    for i in range(T):
        size = .01
        # if i == 0 or i == T - 1:
        #     size = .05
        if i == 0:
            size = .05
        if i == T-1:
            size = .1

        ax.add_patch(plt.Circle(
            (x[2*i:2*i+2]), size, facecolor='none', edgecolor='b'))

    ax.plot()
    plt.show()


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def relu(x):
    return jnp.maximum(0, x)


@jit
def predict_y(params, inputs):
    for W, b in params[:-1]:
        outputs = jnp.dot(W, inputs) + b
        inputs = relu(outputs)
    final_w, final_b = params[-1]
    outputs = jnp.dot(final_w, inputs) + final_b
    return outputs


@functools.partial(jit, static_argnums=(1,))
def full_vec_2_components(input, T):
    L = input[0]
    L_vec = input[1:T]
    x = input[T:3*T]
    delta = input[3*T:3*T+2*(T-1)]
    s = input[3*T+2*(T-1):]
    return L, L_vec, x, delta, s
