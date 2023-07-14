import numpy as np
import logging
import yaml
from jax import vmap
import pandas as pd
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp
import os
import scs
import cvxpy as cp
import jax.scipy as jsp
import jax.random as jra
from l2ws.algo_steps import create_M
from scipy.sparse import csc_matrix
from examples.solve_script import setup_script
from l2ws.launcher import Workspace
from l2ws.algo_steps import get_scaled_vec_and_factor
from jaxopt.projection import projection_simplex
from l2ws.algo_steps import k_steps_train_extragrad, k_steps_eval_extragrad


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
    }
)
log = logging.getLogger(__name__)


def run(run_cfg):
    example = "jamming"
    data_yaml_filename = 'data_setup_copied.yaml'

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    # TODO
    # set the seed
    np.random.seed(setup_cfg['seed'])
    # n_orig = setup_cfg['n_orig']
    # d_mul = setup_cfg['d_mul']
    # k = setup_cfg['k']
    # static_dict = static_canon(n_orig, d_mul, rho_x=rho_x, scale=scale)

    # we directly save q now
    get_q = None
    static_flag = True
    algo = 'extragradient'
    m, n = setup_cfg['n'], setup_cfg['n']
    eg_step = setup_cfg['step_size']
    static_dict = dict(f=jamming_obj, 
                       proj_X=projection_simplex, 
                       proj_Y=projection_simplex, m=m, n=n, eg_step=eg_step)
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example)

    # run the workspace
    workspace.run()


def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    beta_min, beta_max = cfg.beta_min, cfg.beta_max
    sigma_min, sigma_max = cfg.sigma_min, cfg.sigma_max
    n = cfg.n
    k = cfg.solve_iters
    eg_step = cfg.step_size

    np.random.seed(cfg.seed)
    key = jra.PRNGKey(cfg.seed)

    # sample uniformly to get beta, sigma
    beta = beta_min + np.random.rand(N, n) * (beta_max - beta_min)
    sigma = sigma_min + np.random.rand(N, n) * (sigma_max - sigma_min)
    theta_mat = jnp.hstack([beta, sigma])

    # solve each problem using the extragradient method
    proj_X = projection_simplex
    proj_Y = projection_simplex
    
    z0 = jnp.ones(2 * n)
    f = jamming_obj
    # z_final, iter_losses = k_steps_train_extragrad(k, z0, theta_mat[0, :], f, proj_X, proj_Y, n, eg_step,
    #                         supervised=False, z_star=None, jit=True)
    z_stars = jnp.zeros((N, 2 * n))
    for i in range(N):
        z_final, iter_losses, z_all, obj_diffs = k_steps_eval_extragrad(k, z0,
                                                                        theta_mat[i, :],
                                                                        f, proj_X, proj_Y, n, eg_step,
                                                                        supervised=False, z_star=None, jit=True)
        z_stars = z_stars.at[i, :].set(z_final)
        print('fixed point residual', iter_losses[-1])

    # plt.plot(iter_losses)
    # plt.yscale('lo')

    # import pdb
    # pdb.set_trace()
    aa = z_all - z_final
    # diffs = jnp.linalg.norm(aa, axis=1)
    # plt.plot(diffs)
    plt.plot(iter_losses)
    plt.yscale('log')
    plt.savefig("fp_resids.pdf")
    plt.clf()

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    # setup_script(q_mat, theta_mat_jax, solver, data, cones, output_filename, solve=cfg.solve)
    # save the data
    log.info("final saving final data...")
    t0 = time.time()
    jnp.savez(
        output_filename,
        thetas=theta_mat,
        z_stars=z_stars,
    )

    save_time = time.time()
    log.info(f"finished saving final data... took {save_time-t0}'")

    # save plot of first 5 solutions
    for i in range(5):
        plt.plot(z_stars[i, :])
    plt.savefig("z_stars.pdf")
    plt.clf()

    # save plot of first 5 parameters
    for i in range(5):
        plt.plot(theta_mat[i, :])
    plt.savefig("thetas.pdf")
    plt.clf()


def jamming_obj(x, y, theta):
    """
    creates the objective in the saddle problem
    """
    n = x.size
    beta, sigma = theta[:n], theta[n:]

    # costs = jnp.log(1 + beta[i] * x[i] / (sigma[i] + y[i]))
    objs = batch_jamming_costs(beta, sigma, x, y)
    return jnp.sum(objs)


def single_jamming_cost(beta, sigma, x, y):
    return jnp.log(1 + beta * y / (sigma + x))


batch_jamming_costs = vmap(single_jamming_cost, in_axes=(0, 0, 0, 0), out_axes=(0))
