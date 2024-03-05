import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import yaml
from l2ws.launcher import Workspace
from l2ws.examples.solve_script import ista_setup_script
import os
from scipy.sparse import random

def run(run_cfg):
    example = "sparse_coding"
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
    # A = A_scale * jnp.array(np.random.normal(size=(m_orig, n_orig)))

    # evals, evecs = jnp.linalg.eigh(A.T @ A)
    # ista_step =  1 / evals.max()
    # lambd = setup_cfg['lambd']

    # get D
    D = np.random.normal(size=(m_orig, n_orig)) / np.sqrt(m_orig)
    D = D / np.linalg.norm(D, axis=0)
    D = np.array(D)

    

    # get the ista values
    evals, evecs = jnp.linalg.eigh(D.T @ D)
    step = 1 / evals.max()
    lambd = .1
    eta = lambd * step

    # form W
    W = get_W(D)
    W, D = jnp.array(W), jnp.array(D)

    static_dict = dict(D=D, W=W, step=step, eta=eta)

    # we directly save q now
    static_flag = True
    # algo = 'alista'
    algo = run_cfg.get('algo', 'alista')
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example)

    # run the workspace
    workspace.run()


def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    np.random.seed(setup_cfg['seed'])
    m_orig, n_orig = setup_cfg['m_orig'], setup_cfg['n_orig']
    # n2 = int(n_orig / 2)
    # A_scale = setup_cfg['A_scale']
    # # b_scale = setup_cfg['b_scale']
    # A = A_scale * jnp.array(np.random.normal(size=(m_orig, n_orig)))
    
    # evals, evecs = jnp.linalg.eigh(A.T @ A)
    # ista_step = 1 / evals.max()
    # lambd = setup_cfg['lambd']

    np.random.seed(cfg.seed)
    D = np.random.normal(size=(m_orig, n_orig)) / np.sqrt(m_orig)
    D = D / np.linalg.norm(D, axis=0)

    z_orig = np.random.normal(size=(N, n_orig))
    mask = np.random.choice(2, size=(N, n_orig), replace=True, p=[0.9, 0.1])
    z_stars = np.multiply(z_orig, mask)

    if setup_cfg['SNR'] == 'inf':
        noise = 0
    else:
        SNR = setup_cfg['SNR']
        stddev = np.sqrt(np.var(z_stars, axis=1))
        noise_stddev = stddev * np.power (10.0, -SNR / 20.0)
        noise = noise_stddev.reshape(-1, 1) * np.random.normal(size=(N, m_orig))
    b_mat = (D @ z_stars.T).T + noise

    # b_min, b_max = setup_cfg['b_min'], setup_cfg['b_max']
    # # b_mat = b_scale * generate_b_mat(A, N, b_min, b_max)
    # m, n = A.shape
    # b_mat = (b_max - b_min) * np.random.rand(N, m) + b_min

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    # ista_setup_script(b_mat, A, lambd, output_filename)
    jnp.savez(
        output_filename,
        thetas=jnp.array(b_mat),
        z_stars=jnp.array(z_stars),
    )


def get_W(D):
    m, n = D.shape
    W = cp.Variable((m, n))
    obj = cp.norm(W.T @ D, 'fro') ** 2
    constraints = []
    for i in range(m):
        constraints.append(W[i, :] @ D[i, :] == 1)
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS, verbose=True)

    return W.value


def generate_b_mat(A, N, p=.1):
    m, n = A.shape
    b0 = jnp.array(np.random.normal(size=(m)))
    b_mat = 0 * b0 + 1 * jnp.array(np.random.normal(size=(N, m))) + 1
    return b_mat

def eval_ista_obj(z, A, b, lambd):
    return .5 * jnp.linalg.norm(A @ z - b) ** 2 + lambd * jnp.linalg.norm(z, ord=1)


def obj_diff(obj, true_obj):
    return (obj - true_obj)


def sol_2_obj_diff(z, b, true_obj, A, lambd):
    obj = eval_ista_obj(z, A, b, lambd)
    return obj_diff(obj, true_obj)

# def solve_many_probs_cvxpy(A, b_mat, lambd):
#     """
#     solves many lasso problems where each problem has a different b vector
#     """
#     m, n = A.shape
#     N = b_mat.shape[0]
#     z, b_param = cp.Variable(n), cp.Parameter(m)
#     prob = cp.Problem(cp.Minimize(.5 * cp.sum_squares(np.array(A) @ z - b_param) + lambd * cp.norm(z, p=1)))
#     # prob = cp.Problem(cp.Minimize(.5 * cp.sum_squares(np.array(A) @ z - b_param) + lambd * cp.tv(z)))
#     z_stars = jnp.zeros((N, n))
#     objvals = jnp.zeros((N))
#     for i in range(N):
#         b_param.value = np.array(b_mat[i, :])
#         prob.solve(verbose=False)
#         objvals = objvals.at[i].set(prob.value)
#         z_stars = z_stars.at[i, :].set(jnp.array(z.value))
#     print('finished solving cvxpy problems')
#     return z_stars, objvals
