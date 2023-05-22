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
from l2ws.algo_steps import get_scaled_vec_and_factor


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
    }
)
log = logging.getLogger(__name__)


def run(run_cfg):
    example = "sparse_pca"
    data_yaml_filename = 'data_setup_copied.yaml'

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    # set the seed
    np.random.seed(setup_cfg['seed'])
    n_orig = setup_cfg['n_orig']
    k = setup_cfg['k']

    # non-identity DR scaling
    rho_x = run_cfg.get('rho_x', 1)
    scale = run_cfg.get('scale', 1)

    static_dict = static_canon(n_orig, k, rho_x=rho_x, scale=scale)

    # we directly save q now
    get_q = None
    static_flag = True
    workspace = Workspace(run_cfg, static_flag, static_dict, example, get_q)

    # run the workspace
    workspace.run()


def multiple_random_sparse_pca(n_orig, k, r, N, factor=True, seed=42):
    out_dict = static_canon(n_orig, k, factor=factor)
    # c, b = out_dict['c'], out_dict['b']
    P_sparse, A_sparse = out_dict['P_sparse'], out_dict['A_sparse']
    cones = out_dict['cones_dict']
    prob, A_param = out_dict['prob'], out_dict['A_param']
    P, A = jnp.array(P_sparse.todense()), jnp.array(A_sparse.todense())

    # get theta_mat
    A_tensor, theta_mat = generate_A_tensor(N, n_orig, r)
    theta_mat_jax = jnp.array(theta_mat)

    # get theta_mat
    m, n = A.shape
    q_mat = get_q_mat(A_tensor, prob, A_param, m, n)
    # import pdb
    # pdb.set_trace()

    return P, A, cones, q_mat, theta_mat_jax, A_tensor


def generate_A_tensor(N, n_orig, r):
    """
    generates covariance matrices A_1, ..., A_N
        where each A_i has shape (n_orig, n_orig)
    A_i = F Sigma_i F^T
        where F has shape (n_orig, r)
    i.e. each Sigma_i is psd (Sigma_i = B_i B_i^T) and is different
        B_i has shape (r, r)
        F stays the same for each problem
    We let theta = upper_tri(Sigma_i)
    """
    # first generate a random A matrix
    # A0 = np.random.rand(n_orig, n_orig)
    A0 = np.random.normal(size=(n_orig, n_orig))

    # take the SVD
    U, S, VT = np.linalg.svd(A0)

    # take F to be the first r columns of U
    F = U[:, :r]
    A_tensor = np.zeros((N, n_orig, n_orig))
    r_choose_2 = int(r * (r + 1) / 2)
    theta_mat = np.zeros((N, r_choose_2))
    # theta_mat = np.zeros((N, n_orig * r))
    B0 = np.diag(np.sqrt(S[:r]))

    for i in range(N):
        # B = .1*np.random.rand(r, r) #np.diag(np.random.rand(r)) #2 * np.random.rand(r, r) - 1
        # B = np.random.normal(size=(r, r))
        delta = 2 * np.random.rand(r, r) - 1
        B = .1 * delta + B0
        # B = 2 * np.random.rand(r, r)
        # B = np.random.normal(size=(r, r))
        Sigma = .1 * B @ B.T
        col_idx, row_idx = np.triu_indices(r)
        theta_mat[i, :] = Sigma[(row_idx, col_idx)]
        A_tensor[i, :, :] = F @ Sigma @ F.T

        # curr_perturb = np.random.normal(size=(n_orig, r))
        # C = F + .1 * curr_perturb
        # A_tensor[i, :, :] = C @ C.T
        # theta_mat[i, :] = np.ravel(curr_perturb)

    return A_tensor, theta_mat


def cvxpy_prob(n_orig, k):
    A_param = cp.Parameter((n_orig, n_orig), symmetric=True)
    X = cp.Variable((n_orig, n_orig), symmetric=True)
    constraints = [X >> 0, cp.sum(cp.abs(X)) <= k, cp.trace(X) == 1]
    prob = cp.Problem(cp.Minimize(-cp.trace(A_param @ X)), constraints)
    return prob, A_param


def get_q_mat(A_tensor, prob, A_param, m, n):
    N, n_orig, _ = A_tensor.shape
    q_mat = jnp.zeros((N, m + n))
    for i in range(N):
        # set the parameter
        A_param.value = A_tensor[i, :, :]

        # get the problem data
        data, _, __ = prob.get_problem_data(cp.SCS)

        c, b = data['c'], data['b']
        n = c.size
        q_mat = q_mat.at[i, :n].set(c)
        q_mat = q_mat.at[i, n:].set(b)
    return q_mat


def static_canon(n_orig, k, rho_x=1, scale=1, factor=True):
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
    zero_cone_size = cones_cp.zero

    if factor:
        algo_factor, scale_vec = get_scaled_vec_and_factor(M, rho_x, scale, m, n,
                                                           zero_cone_size)
        # algo_factor = jsp.linalg.lu_factor(M + jnp.eye(n + m))
    else:
        algo_factor = None

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

    P, A, cones, q_mat, theta_mat_jax, A_tensor = multiple_random_sparse_pca(
        n_orig, cfg.k, cfg.r, N, factor=False)
    P_sparse, A_sparse = csc_matrix(P), csc_matrix(A)
    m, n = A.shape

    # create scs solver object
    #    we can cache the factorization if we do it like this
    b_np, c_np = np.array(q_mat[0, n:]), np.array(q_mat[0, :n])
    data = dict(P=P_sparse, A=A_sparse, b=b_np, c=c_np)
    tol_abs = cfg.solve_acc_abs
    tol_rel = cfg.solve_acc_rel
    solver = scs.SCS(data, cones, normalize=False, alpha=1, scale=1,
                     rho_x=1, adaptive_scale=False, eps_abs=tol_abs, eps_rel=tol_rel)

    setup_script(q_mat, theta_mat_jax, solver, data, cones, output_filename, solve=cfg.solve)

    import pdb
    pdb.set_trace()
