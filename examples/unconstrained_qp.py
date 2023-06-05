import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import yaml
from l2ws.launcher import Workspace

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

    Q = jnp.array(np.random.normal(size=(n_orig, n_orig)))
    P = Q @ Q.T
    evals, evecs = jnp.linalg.eigh(P)
    gd_step = 1 / evals.max()

    # static_dict = static_canon(n_orig, k, rho_x=rho_x, scale=scale)
    static_dict = dict(Q=Q, gd_step=gd_step)

    # we directly save q now
    get_q = None
    static_flag = True
    algo = 'gd'
    workspace = Workspace(run_cfg, algo, static_flag, static_dict, example, get_q)

    # run the workspace
    workspace.run()

# def generate_b_mat(A, N):
#     m, n = A.shape
#     b_mat = jnp.zeros((N, m))
#     x_star_mask = np.random.binomial(1, p, size=(N, n))
#     x_stars_dense = jnp.array(np.random.normal(size=(N, n)))
#     x_stars = jnp.multiply(x_star_mask, x_stars_dense)
#     for i in range(N):
#         b = A @ x_stars[i, :]
#         b_mat = b_mat.at[i, :].set(b)
#     return b_mat

# def eval_ista_obj(z, A, b, lambd):
#     return .5 * jnp.linalg.norm(A @ z - b) ** 2 + lambd * jnp.linalg.norm(z, ord=1)

def obj(Q, c, z):
    return .5 * z.T @ Q @ z  + c @ z

def obj_diff(obj, true_obj):
    return (obj - true_obj)


# def sol_2_obj_diff(z, b, true_obj, A, lambd):
#     obj = eval_ista_obj(z, A, b, lambd)
#     return obj_diff(obj, true_obj)

def solve_many_probs_cvxpy(Q, c_mat):
    """
    solves many unconstrained qp problems where each problem has a different b vector
    """
    # m, n = A.shape
    Q_inv = jnp.linalg.inv(Q)
    z_stars = -Q_inv @ c_mat
    objvals = obj(Q, c_mat, z_stars)
    return z_stars, objvals
