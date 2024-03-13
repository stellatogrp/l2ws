import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import yaml
from l2ws.launcher import Workspace
from l2ws.examples.solve_script import gd_setup_script
import os


def run(run_cfg):
    example = "unconstrained_qp"
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

    split = setup_cfg['P_split']
    split_factor = setup_cfg['split_factor']
    P_vec = jnp.concatenate([split_factor * jnp.ones(split), 1 * jnp.ones(n_orig - split)])
    P = jnp.diag(P_vec)

    # A_scale = setup_cfg['A_scale']
    # A = A_scale * jnp.array(np.random.normal(size=(m_orig, n_orig)))
    # split = int(n_orig / 2)
    # A_vec = jnp.concatenate([100 * jnp.ones(split), 1 * jnp.ones(split)])
    # A = jnp.diag(A_vec)
    # density = 0.1
    # A = A_scale * jnp.array(random(m_orig, n_orig, density=density, format='csr').todense())
    # evals, evecs = jnp.linalg.eigh(A.T @ A)
    # gd_step =  1 / evals.max()
    # gd_step = 1 / P.max() # 2 / (P.max() +  1) #1 / P.max()
    gd_step = 1 / P.max() #2 / (P.max() +  1) #1 / P.max()

    static_dict = dict(P=P, gd_step=gd_step)

    # we directly save q now
    static_flag = True
    algo = 'gd'
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example)

    # run the workspace
    workspace.run()


def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    np.random.seed(setup_cfg['seed'])
    n_orig = setup_cfg['n_orig']

    split = setup_cfg['P_split']
    split_factor = setup_cfg['split_factor']
    P_vec = jnp.concatenate([split_factor * jnp.ones(split), 1 * jnp.ones(n_orig - split)])
    P = jnp.diag(P_vec)
    c_min = setup_cfg['c_min']
    c_max = setup_cfg['c_max']
    # range_factor = setup_cfg['range_factor']


    """
    P is a diagonal matrix diag(p_1, ..., p_n)
    p_1, ..., p_split = split_factor
    p_{split+1}, ..., n = 1
    sample c in the following way
    c_i = p_i b_i cos(theta pi i / n)

    b is range_factor
    """

    # generate theta
    
    # generate c_mat

    c1_mat = split_factor ** 2 * (c_min + (c_max - c_min) * jnp.array(np.random.rand(N, split)))
    c2_mat = c_min + (c_max - c_min) * jnp.array(np.random.rand(N, n_orig - split))
    c_mat = jnp.hstack([c1_mat, c2_mat])

    np.random.seed(cfg.seed)

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    # b_min, b_max = setup_cfg['b_min'], setup_cfg['b_max']
    # b_mat = b_scale * generate_b_mat(A, N, b_min, b_max)
    # m, n = A.shape
    # b_mat = (b_max - b_min) * np.random.rand(N, m) + b_min

    gd_setup_script(c_mat, P, output_filename)


# def generate_b_mat(A, N, p=.1):
#     m, n = A.shape
#     b0 = jnp.array(np.random.normal(size=(m)))
#     b_mat = 0 * b0 + 1 * jnp.array(np.random.normal(size=(N, m))) + 1
#     # b_mat = jnp.zeros((N, m))
#     # x_star_mask = np.random.binomial(1, p, size=(N, n))
#     # x_stars_dense = jnp.array(np.random.normal(size=(N, n)))
#     # x_stars = jnp.multiply(x_star_mask, x_stars_dense)
#     # for i in range(N):
#     #     b = A @ x_stars[i, :]
#     #     b_mat = b_mat.at[i, :].set(b)
#     return b_mat

# def eval_gd_obj(z, A, b, lambd):
#     return .5 * jnp.linalg.norm(A @ z - b) ** 2 + lambd * jnp.linalg.norm(z, ord=1)


# def obj_diff(obj, true_obj):
#     return (obj - true_obj)


# def sol_2_obj_diff(z, b, true_obj, A, lambd):
#     obj = eval_gd_obj(z, A, b, lambd)
#     return obj_diff(obj, true_obj)

# def solve_many_probs_cvxpy(A, b_mat, lambd):
#     """
#     solves many gd problems where each problem has a different b vector
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


# def run(run_cfg):
#     example = "unconstrained_qp"
#     data_yaml_filename = 'data_setup_copied.yaml'

#     # read the yaml file
#     with open(data_yaml_filename, "r") as stream:
#         try:
#             setup_cfg = yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#             setup_cfg = {}

#     # set the seed
#     np.random.seed(setup_cfg['seed'])
#     n_orig = setup_cfg['n_orig']

#     Q = jnp.array(np.random.normal(size=(n_orig, n_orig)))
#     P = Q @ Q.T
#     evals, evecs = jnp.linalg.eigh(P)
#     gd_step = 1 / evals.max()

#     # static_dict = static_canon(n_orig, k, rho_x=rho_x, scale=scale)
#     static_dict = dict(Q=Q, gd_step=gd_step)

#     # we directly save q now
#     get_q = None
#     static_flag = True
#     algo = 'gd'
#     workspace = Workspace(run_cfg, algo, static_flag, static_dict, example, get_q)

#     # run the workspace
#     workspace.run()


def obj(Q, c, z):
    return .5 * z.T @ Q @ z  + c @ z

def obj_diff(obj, true_obj):
    return (obj - true_obj)


def solve_many_probs_cvxpy(Q, c_mat):
    """
    solves many unconstrained qp problems where each problem has a different b vector
    """
    # m, n = A.shape
    Q_inv = jnp.linalg.inv(Q)
    z_stars = -Q_inv @ c_mat
    objvals = obj(Q, c_mat, z_stars)
    return z_stars, objvals
