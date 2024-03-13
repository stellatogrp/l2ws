import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import yaml
from l2ws.launcher import Workspace
from l2ws.examples.solve_script import ista_setup_script
import os
from scipy.sparse import random

def run(run_cfg):
    example = "lasso"
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
    m_orig, n_orig = setup_cfg['m_orig'], setup_cfg['n_orig']
    A_scale = setup_cfg['A_scale']
    A = A_scale * jnp.array(np.random.normal(size=(m_orig, n_orig)))
    # n2 = int(n_orig / 2)
    # A = A.at[:, :n2].set(A[:, :n2] / 10)
    # A = A.at[:, :(int(n_orig / 2))] * 100
    # split = int(n_orig / 2)
    # A_vec = jnp.concatenate([100 * jnp.ones(split), 1 * jnp.ones(split)])
    # A = jnp.diag(A_vec)
    # density = 0.1
    # A = A_scale * jnp.array(random(m_orig, n_orig, density=density, format='csr').todense())
    evals, evecs = jnp.linalg.eigh(A.T @ A)
    ista_step =  1 / evals.max()
    lambd = setup_cfg['lambd']

    static_dict = dict(A=A, lambd=lambd, ista_step=ista_step)

    # we directly save q now
    static_flag = True
    algo = 'ista'
    

    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example)

    

    # run the workspace
    workspace.run()


def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    np.random.seed(setup_cfg['seed'])
    m_orig, n_orig = setup_cfg['m_orig'], setup_cfg['n_orig']
    n2 = int(n_orig / 2)
    A_scale = setup_cfg['A_scale']
    # b_scale = setup_cfg['b_scale']
    A = A_scale * jnp.array(np.random.normal(size=(m_orig, n_orig)))
    # A = A.at[:, :n2].set(A[:, :n2] / 10)
    # split = int(n_orig / 2)
    # A_vec = jnp.concatenate([100 * jnp.ones(split), 1 * jnp.ones(split)])
    # A = jnp.diag(A_vec)
    # density = 0.1
    # A = A_scale * jnp.array(random(m_orig, n_orig, density=density, format='csr').todense())
    evals, evecs = jnp.linalg.eigh(A.T @ A)
    ista_step = 1 / evals.max()
    lambd = setup_cfg['lambd']

    np.random.seed(cfg.seed)

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    b_min, b_max = setup_cfg['b_min'], setup_cfg['b_max']
    # b_mat = b_scale * generate_b_mat(A, N, b_min, b_max)
    m, n = A.shape
    b_mat = (b_max - b_min) * np.random.rand(N, m) + b_min

    # 
    # b_mat[:, :n2] = b_mat[:, :n2] / 10

    ista_setup_script(b_mat, A, lambd, output_filename)


def generate_b_mat(A, N, p=.1):
    m, n = A.shape
    b0 = jnp.array(np.random.normal(size=(m)))
    b_mat = 0 * b0 + 1 * jnp.array(np.random.normal(size=(N, m))) + 1
    # b_mat = jnp.zeros((N, m))
    # x_star_mask = np.random.binomial(1, p, size=(N, n))
    # x_stars_dense = jnp.array(np.random.normal(size=(N, n)))
    # x_stars = jnp.multiply(x_star_mask, x_stars_dense)
    # for i in range(N):
    #     b = A @ x_stars[i, :]
    #     b_mat = b_mat.at[i, :].set(b)
    return b_mat

def eval_ista_obj(z, A, b, lambd):
    return .5 * jnp.linalg.norm(A @ z - b) ** 2 + lambd * jnp.linalg.norm(z, ord=1)


def obj_diff(obj, true_obj):
    return (obj - true_obj)


def sol_2_obj_diff(z, b, true_obj, A, lambd):
    obj = eval_ista_obj(z, A, b, lambd)
    return obj_diff(obj, true_obj)

def solve_many_probs_cvxpy(A, b_mat, lambd):
    """
    solves many lasso problems where each problem has a different b vector
    """
    m, n = A.shape
    N = b_mat.shape[0]
    z, b_param = cp.Variable(n), cp.Parameter(m)
    prob = cp.Problem(cp.Minimize(.5 * cp.sum_squares(np.array(A) @ z - b_param) + lambd * cp.norm(z, p=1)))
    # prob = cp.Problem(cp.Minimize(.5 * cp.sum_squares(np.array(A) @ z - b_param) + lambd * cp.tv(z)))
    z_stars = jnp.zeros((N, n))
    objvals = jnp.zeros((N))
    for i in range(N):
        b_param.value = np.array(b_mat[i, :])
        prob.solve(verbose=False)
        objvals = objvals.at[i].set(prob.value)
        z_stars = z_stars.at[i, :].set(jnp.array(z.value))
    print('finished solving cvxpy problems')
    return z_stars, objvals
