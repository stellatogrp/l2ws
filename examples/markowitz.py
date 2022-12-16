import hydra
import cvxpy as cp
from jax import linear_transpose, vmap
import pandas as pd
from pandas import read_csv
from l2ws.scs_problem import SCSinstance, scs_jax
import numpy as np
import pdb
from l2ws.launcher import Workspace
from scipy import sparse
import jax.numpy as jnp
from scipy.sparse import coo_matrix, bmat, csc_matrix
import jax.scipy as jsp
import time
import matplotlib.pyplot as plt
import os
import scs
import logging
import yaml
# SCALE_FACTOR = 1e2
log = logging.getLogger(__name__)


def run(run_cfg):
    '''
    retrieve data for this config
    theta is all of the following
    theta = (ret, pen_risk, pen_hold, pen_trade, w0)

    Sigma is constant

     just need (theta, factor, u_star), Pi
    '''
    # todo: retrieve data and put into a nice form - OR - just save to nice form

    '''
    create workspace
    needs to know the following somehow -- from the run_cfg
    1. nn cfg
    2. (theta, factor, u_star)_i=1^N
    3. Pi

    2. and 3. are stored in data files and the run_cfg holds the location

    it will create the l2a_model
    '''
    
   
    datetime = run_cfg.data.datetime
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'markowitz'
    data_yaml_filename = f"{orig_cwd}/outputs/{example}/aggregate_outputs/{datetime}/data_setup_copied.yaml"

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    pen_ret = 10**setup_cfg['pen_rets_min']
    a = setup_cfg['a']
    static_dict = static_canon(setup_cfg['data'], a, setup_cfg['idio_risk'], setup_cfg['scale_factor'])
    def get_q(theta):
        q = jnp.zeros(2*a + 1)
        q = q.at[:a].set(-theta * pen_ret)
        q = q.at[a].set(1)
        return q
    get_q_batch = vmap(get_q, in_axes=(0), out_axes=(0))
    static_flag = True
    workspace = Workspace(run_cfg, static_flag, static_dict, 'markowitz', get_q_batch)

    '''
    run the workspace
    '''
    workspace.run()


def setup_probs(setup_cfg):
    print('entered convex markowitz', flush=True)
    cfg = setup_cfg

    a = cfg.a
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    # std_mult = cfg.std_mult
    pen_rets_min = cfg.pen_rets_min
    pen_rets_max = cfg.pen_rets_max
    alpha = cfg.alpha
    # max_clip, min_clip = cfg.max_clip, cfg.min_clip

    # p is the size of each feature vector (mu and pen_rets_factor)
    if pen_rets_max > pen_rets_min:
        p = a + 1
    else:
        p = a
    thetas = jnp.zeros((N, p))

    # read in the returns dataframe
    orig_cwd = hydra.utils.get_original_cwd()
    if cfg.data == 'yahoo':
        ret_cov_np = f"{orig_cwd}/data/portfolio_data/yahoo_ret_cov.npz"
    elif cfg.data == 'nasdaq':
        ret_cov_np = f"{orig_cwd}/data/portfolio_data/ret_cov.npz"
    elif cfg.data == 'eod':
        ret_cov_np = f"{orig_cwd}/data/portfolio_data/eod_ret_cov_factor.npz"
    
    ret_cov_loaded = np.load(ret_cov_np)
    Sigma = ret_cov_loaded['cov'] + np.eye(a) * cfg.idio_risk
    ret = ret_cov_loaded['ret'][1:, :a]

    
    # ret_mean = ret.mean(axis=0)
    # clipped_ret_mean_orig = np.clip(ret_mean, a_min=min_clip, a_max=max_clip)
    # clipped_ret_mean = clipped_ret_mean_orig * SCALE_FACTOR

    log.info('creating static canonicalization...')
    t0 = time.time()
    out_dict = static_canon(cfg.data, a, cfg.idio_risk, cfg.scale_factor)
    t1 = time.time()
    log.info(f"finished static canonicalization - took {t1-t0} seconds")

    A_sparse = out_dict['A_sparse']
    A_sparse, P_sparse = out_dict['A_sparse'], out_dict['P_sparse']
    b = out_dict['b']
    n = a
    m = a + 1
    cones_dict = dict(z=1, l=a)

    '''
    save output to output_filename
    '''
    # save to outputs/mm-dd-ss/... file
    if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
        slurm_idx = os.environ["SLURM_ARRAY_TASK_ID"]
        output_filename = f"{os.getcwd()}/data_setup_slurm_{slurm_idx}"
    else:
        output_filename = f"{os.getcwd()}/data_setup_slurm"
    '''
    create scs solver object
    we can cache the factorization if we do it like this
    '''
    blank_b = np.zeros(m)
    blank_c = np.zeros(n)
    data = dict(P=P_sparse, A=A_sparse, b=blank_b, c=blank_c)
    tol = cfg.solve_acc
    solver = scs.SCS(data, cones_dict, eps_abs=tol, eps_rel=tol)
    solve_times = np.zeros(N)
    x_stars = jnp.zeros((N, n))
    y_stars = jnp.zeros((N, m))
    s_stars = jnp.zeros((N, m))
    q_mat = jnp.zeros((N, n + m))
    mu_mat = np.zeros((N, a))
    pen_rets = np.zeros(N)
    scs_instances = []

    # mu_mat = SCALE_FACTOR * np.random.multivariate_normal(clipped_ret_mean_orig, Sigma, size=(N))
    # pdb.set_trace()
    mu_mat = np.zeros((N, a))
    T = ret.shape[0]
    noise = np.sqrt(.02) * np.random.normal(size=(N, a))
    
    for i in range(N):
        log.info(f"solving problem number {i}")
        time_index = i % T
        mu_mat[i, :] = cfg.scale_factor * alpha * (ret[time_index, :] + noise[i, :])
        # mu_mat[i, :] = clipped_ret_mean * \
        #     (1 + std_mult*np.random.normal(size=(a))
        #      ) + std_mult *np.random.normal(size=(a))

        sample = np.random.rand(1) * (pen_rets_max -
                                      pen_rets_min) + pen_rets_min
        pen_rets[i] = 10 ** sample
        thetas = thetas.at[i, :a].set(mu_mat[i, :])
        if pen_rets_max > pen_rets_min:
            thetas = thetas.at[i, a].set(pen_rets[i])

        # manual canon
        c = -mu_mat[i, :] * pen_rets[i]
        manual_canon_dict = {'P': P_sparse, 'A': A_sparse,
                             'b': b, 'c': c,
                             'cones': cones_dict}
        scs_instance = SCSinstance(
            manual_canon_dict, solver, manual_canon=True)

        scs_instances.append(scs_instance)
        x_stars = x_stars.at[i, :].set(scs_instance.x_star)
        y_stars = y_stars.at[i, :].set(scs_instance.y_star)
        s_stars = s_stars.at[i, :].set(scs_instance.s_star)
        q_mat = q_mat.at[i, :].set(scs_instance.q)
        solve_times[i] = scs_instance.solve_time

        # input into our_scs
        # P_jax, A_jax = jnp.array(P_sparse.todense()), jnp.array(A_sparse.todense())
        # b_jax, c_jax = jnp.array(b), jnp.array(c)
        # data = dict(P=P_jax, A=A_jax, b=b_jax, c=c_jax, cones=cones_dict)
        # scs_jax(data, iters=1000)
        # pdb.set_trace()
    
    # resave the data??
    # print('saving final data...', flush=True)
    log.info('saving final data...')
    t0 = time.time()
    jnp.savez(output_filename,
              thetas=thetas,
              x_stars=x_stars,
              y_stars=y_stars,
              )
    save_time = time.time()
    log.info(f"finished saving final data... took {save_time-t0}'")

    # save plot of first 5 solutions
    for i in range(20):
        plt.plot(x_stars[i, :], label=i)
    if T < N:
        plt.plot(x_stars[T, :], label=T)
    plt.legend()
    plt.savefig('opt_solutions.pdf')
    plt.clf()

    # save plot of first 5 parameters
    for i in range(5):
        plt.plot(thetas[i, :], label=i)
    if T < N:
        plt.plot(thetas[T, :], label=T)
    plt.legend()
    plt.savefig('thetas.pdf')



def static_canon(data, a, idio_risk, scale_factor):
    '''
    This method produces the parts of each problem that does not change
    i.e. P, A, b, cones

    It creates the matrix 
    M = [P A.T
        -A 0]

    It also returns the necessary factorizations
    1. factor(I + M)
    2. factor(A.T A)
    '''
    orig_cwd = hydra.utils.get_original_cwd()
    if data == 'yahoo':
        ret_cov_np = f"{orig_cwd}/data/portfolio_data/yahoo_ret_cov.npz"
    elif data == 'nasdaq':
        ret_cov_np = f"{orig_cwd}/data/portfolio_data/ret_cov.npz"
    elif data == 'eod':
        ret_cov_np = f"{orig_cwd}/data/portfolio_data/eod_ret_cov_factor.npz"
    
    ret_cov_loaded = np.load(ret_cov_np)
    Sigma = ret_cov_loaded['cov'][:a, :a] + idio_risk * np.eye(a)
    n = a
    m = a + 1

    # scale Sigma
    Sigma = scale_factor * Sigma

    # do the manual canonicalization
    b = np.zeros(a + 1)
    b[0] = 1
    A = np.zeros((a + 1, a))
    A[0, :] = 1
    A[1:, :] = -np.eye(a)
    A_sparse = csc_matrix(A)
    P_sparse = csc_matrix(Sigma)

    # cones
    cones_dict = dict(z=1, l=a)
    cones_array = jnp.array([cones_dict['z'], cones_dict['l']])

    # factor for dual prediction from primal
    ATA_factor = jsp.linalg.cho_factor(A.T @ A)

    # create the matrix M
    M = jnp.zeros((n + m, n + m))
    P_jax = jnp.array(Sigma)
    A_jax = jnp.array(A)
    M = M.at[:n, :n].set(P_jax)
    M = M.at[:n, n:].set(A_jax.T)
    M = M.at[n:, :n].set(-A_jax)

    # factor for DR splitting
    algo_factor = jsp.linalg.lu_factor(M + jnp.eye(n+m))

    out_dict = dict(Sigma=Sigma, M=M,
                    ATA_factor=ATA_factor,
                    algo_factor=algo_factor,
                    cones_array=cones_array,
                    A_sparse=A_sparse,
                    P_sparse=P_sparse,
                    b=b)
    return out_dict


'''
code for automatic canon
# cvxpy problem for automatic canon
#  S = np.linalg.cholesky(Sigma)
# x = cp.Variable(a)
# lin_param = cp.Parameter(a)
# constraints = [cp.sum(x) == 1, x >= 0]
# prob = cp.Problem(
#     cp.Minimize(.5*cp.sum_squares(S.T @ x) - lin_param @ x), constraints)

# automatic canon
# lin_param.value = mu_mat[i, :] * pen_rets[i]
# scs_instance = SCSinstance(prob, manual_canon=False)

# get necessary sizes
# automatic
# m, n = prob.get_problem_data(cp.SCS)[0]['A'].shape
'''

def our_scs():
    a = 3000
    m, n = a+1, a
    # Sigma, M = out_dict['Sigma'], out_dict['M']
    # ATA_factor, algo_factor = out_dict['ATA_factor'], out_dict['algo_factor']
    # cones_array, A_sparse = out_dict['cones_array'], out_dict['A_sparse']
    # A_sparse, P_sparse = out_dict['A_sparse'], out_dict['P_sparse']
    # b = out_dict['b']
    # n = a
    # m = a + 1
    # cones_dict = dict(z=1, l=a)
    # pdb.set_trace()

    ret_cov_np = f"data/portfolio_data/ret_cov.npz"
    #ret_cov_np = f"{orig_cwd}/data/portfolio_data/ret_cov.npz"
    ret_cov_loaded = np.load(ret_cov_np)
    P = ret_cov_loaded['cov'][:a, :a] + 1e-6 * np.eye(a)
    ret = ret_cov_loaded['ret'][:, :a]
    A = np.zeros((a + 1, a))
    A[0, :] = 1
    A[1:, :] = -np.eye(a)
    b = jnp.zeros(m)
    b = b.at[0].set(1)
    cones_dict = dict(z=1, l=a)

    # input into our_scs
    P_jax, A_jax = jnp.array(P), jnp.array(A)
    b_jax, c_jax = jnp.array(b), jnp.array(ret[0, :])
    data = dict(P=P_jax*1000, A=A_jax, b=b_jax, c=c_jax*1000*.03, cones=cones_dict)
    scs_jax(data)


if __name__ == '__main__':
    our_scs()
