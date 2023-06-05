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
    np.random.seed(setup_cfg['seed'])
    n_orig = setup_cfg['n_orig']
    d_mul = setup_cfg['d_mul']
    # k = setup_cfg['k']
    

    # non-identity DR scaling
    rho_x = run_cfg.get('rho_x', 1)
    scale = run_cfg.get('scale', 1)

    static_dict = static_canon(n_orig, d_mul, rho_x=rho_x, scale=scale)

    # we directly save q now
    get_q = None
    static_flag = True
    algo = 'scs'
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example)

    # run the workspace
    workspace.run()


def multiple_random_phase_retrieval(n_orig, d_mul, x_mean, x_var, N, seed=42):
    ######################### TODO
    out_dict = static_canon(n_orig, d_mul)
    # # c, b = out_dict['c'], out_dict['b']
    P_sparse, A_sparse = out_dict['P_sparse'], out_dict['A_sparse']
    cones = out_dict['cones_dict']
    prob, b_param = out_dict['prob'], out_dict['b_param']
    P, A = jnp.array(P_sparse.todense()), jnp.array(A_sparse.todense())
    A_tensor = out_dict['A_tensor']


    # get theta_mat and b_vals together
    b_matrix = generate_theta_mat_b_vals(N, A_tensor, x_mean, x_var, n_orig, d_mul)
    theta_mat_jax = jnp.array(b_matrix)

    # convert to q_mat
    m, n = A.shape
    q_mat = get_q_mat(b_matrix, prob, b_param, m, n)

    return P, A, cones, q_mat, theta_mat_jax # possibly return more


def generate_psi_vecs(n, d):
    '''
        Generate vector of psi's, TODO: type up details 
    '''
    # first random var
    A_vals = np.array([1, 1j, -1, -1j])
    psi_A = np.random.choice(A_vals, size=(n, d))
    # print('psi_A', psi_A)
    
    # second random var
    B_vals = np.array([np.sqrt(2)/2, np.sqrt(3)])
    B_probs = np.array([0.2, 0.8])
    psi_B = np.random.choice(B_vals, size=(n, d), p=B_probs)
    # print('psi_B', psi_B)

    # element-wise multiply
    out = np.multiply(psi_A, psi_B, dtype='complex_')
    return out


def generate_A_tensor(n_orig, d_mul):
    dftmtx = jnp.fft.fft(jnp.eye(n_orig))
    d = n_orig * d_mul
    A_out = np.zeros((d, n_orig, n_orig), dtype='complex_')
    psi = generate_psi_vecs(n_orig, d_mul)
    for l in range(n_orig):
        Wl = dftmtx[l, :].reshape(1, -1)
        for j in range(d_mul):
            curr_psi = psi[:, j]
            ai = np.multiply(Wl, curr_psi.conjugate(), dtype='complex_')
            Ai = ai.T @ ai.conjugate()
            A_out[(j-1) * n_orig + l, :, :] = Ai 
    return A_out / 100


def cvxpy_prob(n_orig, d_mul, seed=42):
    """
    TODO adapt for phase retrieval
    will need to pass in specific A_i matrices
    """
    d = n_orig * d_mul
    A_tensor = generate_A_tensor(n_orig, d_mul)

    ####### this was for sparse pca
    # A_param = cp.Parameter((n_orig, n_orig), symmetric=True)
    # X = cp.Variable((n_orig, n_orig), symmetric=True)
    # constraints = [X >> 0, cp.sum(cp.abs(X)) <= k, cp.trace(X) == 1]
    # prob = cp.Problem(cp.Minimize(-cp.trace(A_param @ X)), constraints)
    # return prob, A_param

    b_param = cp.Parameter(d)
    X = cp.Variable((n_orig, n_orig), hermitian=True)
    constraints = [X >> 0]
    for i in range(d):
        Ai = A_tensor[i]
        constraints += [cp.trace(Ai @ X) == b_param[i]]
    prob = cp.Problem(cp.Minimize(cp.trace(X)), constraints)
    return prob, A_tensor, b_param


def generate_theta_mat_b_vals(N, A_tensor, x_mean, x_var, n_orig, d_mul):
    d = n_orig * d_mul
    b_matrix = np.zeros((N, d))
    # n_orig_choose_2 = int(n_orig * (n_orig + 1) / 2)
    # theta_mat = np.zeros((N, n_orig_choose_2), dtype='complex_')
    for i in range(N):
        # this is where the parameterization comes in
        # could modify where the xi comes from
        negate1 = np.random.binomial(n=1, p=0.5, size=(n_orig))
        negate2 = np.random.binomial(n=1, p=0.5, size=(n_orig))
        negate1[negate1 == 0] = -1
        negate2[negate2 == 0] = -1

        xi = np.multiply(np.random.normal(size=(n_orig), loc=x_mean, scale=np.sqrt(x_var)), negate1) \
            + 1j * np.multiply(np.random.normal(size=(n_orig), loc=x_mean, scale=np.sqrt(x_var)), negate2)
        Xi = np.outer(xi, xi.conjugate())
        # col_idx, row_idx = np.triu_indices(n_orig)
        # theta_mat[i, :] = Xi[(col_idx, row_idx)]
        # import pdb
        # pdb.set_trace()
        for j in range(d):
            # the trace will be real for hermitian matrices, but we use np.real to remove small complex floats
            b_matrix[i, j] = np.real(np.trace(A_tensor[j] @ Xi))
    return b_matrix


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


def static_canon(n_orig, d_mul, rho_x=1, scale=1, factor=True, seed=42):
    # create the cvxpy problem
    prob, A_tensor, b_param = cvxpy_prob(n_orig, d_mul, seed=42)

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

    # import pdb
    # pdb.set_trace()

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
        # A_param=A_param,
        A_tensor=A_tensor,
        b_param=b_param,
    )
    return out_dict


def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    n_orig = cfg.n_orig
    d_mul = cfg.d_mul
    x_var = cfg.x_var
    x_mean = cfg.x_mean
    # d_orig = n_orig * d_mul

    np.random.seed(cfg.seed)
    key = jra.PRNGKey(cfg.seed)

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    ################## TODO add extra params to generation
    P, A, cones, q_mat, theta_mat_jax = multiple_random_phase_retrieval(
        n_orig, d_mul, x_mean, x_var, N) 

    P_sparse, A_sparse = csc_matrix(P), csc_matrix(A)
    m, n = A.shape

    # create scs solver object
    #    we can cache the factorization if we do it like this
    b_np, c_np = np.array(q_mat[0, n:]), np.array(q_mat[0, :n])
    data = dict(P=P_sparse, A=A_sparse, b=b_np, c=c_np)
    tol_abs = cfg.solve_acc_abs
    tol_rel = cfg.solve_acc_rel
    max_iters = cfg.get('solve_max_iters', 10000)
    solver = scs.SCS(data, cones, eps_abs=tol_abs, eps_rel=tol_rel, max_iters=max_iters)

    setup_script(q_mat, theta_mat_jax, solver, data, cones, output_filename, solve=cfg.solve)
