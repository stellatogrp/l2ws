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
from l2ws.algo_steps import create_M
from scipy.sparse import csc_matrix
from examples.solve_script import setup_script
from l2ws.launcher import Workspace


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
    }
)
log = logging.getLogger(__name__)


def run(run_cfg):
    example = "phase_retrieval"
    data_yaml_filename = 'data_setup_copied.yaml'

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    ######################### TODO
    # set the seed
    # np.random.seed(setup_cfg['seed'])
    # n_orig = setup_cfg['n_orig']
    # k = setup_cfg['k']
    # static_dict = static_canon(n_orig, k)


    # we directly save q now
    get_q = None
    static_flag = True
    workspace = Workspace(run_cfg, static_flag, static_dict, example, get_q)

    # run the workspace
    workspace.run()


def multiple_random_phase_retrieval():
    ######################### TODO
    # out_dict = static_canon(n_orig, k)
    # # c, b = out_dict['c'], out_dict['b']
    # P_sparse, A_sparse = out_dict['P_sparse'], out_dict['A_sparse']
    # cones = out_dict['cones_dict']
    # prob, A_param = out_dict['prob'], out_dict['A_param']
    # P, A = jnp.array(P_sparse.todense()), jnp.array(A_sparse.todense())

    # # get theta_mat
    # A_tensor, theta_mat = generate_A_tensor(N, n_orig, r)
    # theta_mat_jax = jnp.array(theta_mat)

    # # get theta_mat
    # m, n = A.shape
    # q_mat = get_q_mat(A_tensor, prob, A_param, m, n)

    return P, A, cones, q_mat, theta_mat_jax # possibly return more


def cvxpy_prob(n_orig, k):
    """
    TODO adapt for phase retrieval
    will need to pass in specific A_i matrices
    """

    ####### this was for sparse pca
    # A_param = cp.Parameter((n_orig, n_orig), symmetric=True)
    # X = cp.Variable((n_orig, n_orig), symmetric=True)
    # constraints = [X >> 0, cp.sum(cp.abs(X)) <= k, cp.trace(X) == 1]
    # prob = cp.Problem(cp.Minimize(-cp.trace(A_param @ X)), constraints)
    # return prob, A_param

    return prob, b_param


def get_q_mat(b_matrix, prob, b_param, m, n):
    """
    change this so that b_matrix, b_param is passed in
        instead of A_tensor, A_param

    I think this should work now
    """
    N = b_matrix.shape[0]
    q_mat = jnp.zeros((N, m + n))
    for i in range(N):
        # set the parameter
        b_param.value = b_matrix[i, :]

        # get the problem data
        data, _, __ = prob.get_problem_data(cp.SCS)

        c, b = data['c'], data['b']
        n = c.size
        q_mat = q_mat.at[i, :n].set(c)
        q_mat = q_mat.at[i, n:].set(b)
    return q_mat


def static_canon(n_orig, k):
    # create the cvxpy problem
    prob, A_param = cvxpy_prob(n_orig, k)

    # get the problem data
    data, _, __ = prob.get_problem_data(cp.SCS)

    A_sparse, c, b = data['A'], data['c'], data['b']
    m, n = A_sparse.shape
    P_sparse = csc_matrix(np.zeros((n, n)))
    cones_cp = data['dims']

    # factor for DR splitting
    m, n = A_sparse.shape
    P_jax, A_jax = jnp.array(P_sparse.todense()), jnp.array(A_sparse.todense())
    M = create_M(P_jax, A_jax)
    algo_factor = jsp.linalg.lu_factor(M + jnp.eye(n + m))

    # set the dict
    cones = {'z': cones_cp.zero, 'l': cones_cp.nonneg, 'q': cones_cp.soc, 's': cones_cp.psd}
    out_dict = dict(
        M=M,
        algo_factor=algo_factor,
        cones_dict=cones,
        A_sparse=A_sparse,
        P_sparse=P_sparse,
        b=b,
        c=c,
        prob=prob,
        A_param=A_param
    )
    return out_dict


def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    n_orig = cfg.n_orig

    np.random.seed(cfg.seed)

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    ################## TODO something like this
    # P, A, cones, q_mat, theta_mat_jax = multiple_random_phase_retrieval(inputs)

    P_sparse, A_sparse = csc_matrix(P), csc_matrix(A)
    m, n = A.shape

    # create scs solver object
    #    we can cache the factorization if we do it like this
    b_np, c_np = np.array(q_mat[0, n:]), np.array(q_mat[0, :n])
    data = dict(P=P_sparse, A=A_sparse, b=b_np, c=c_np)
    tol_abs = cfg.solve_acc_abs
    tol_rel = cfg.solve_acc_rel
    solver = scs.SCS(data, cones, eps_abs=tol_abs, eps_rel=tol_rel)

    setup_script(q_mat, theta_mat_jax, solver, data, cones, output_filename)
