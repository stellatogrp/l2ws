import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import yaml
from l2ws.launcher import Workspace

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
    n_orig = setup_cfg['n_orig']
    lambd = setup_cfg['lambd']
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

def generate_b_mat(A, N, p=.1):
    m, n = A.shape
    b_mat = jnp.zeros((N, m))
    x_star_mask = np.random.binomial(1, p, size=(N, n))
    x_stars_dense = jnp.array(np.random.normal(size=(N, n)))
    x_stars = jnp.multiply(x_star_mask, x_stars_dense)
    for i in range(N):
        b = A @ x_stars[i, :]
        b_mat = b_mat.at[i, :].set(b)
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
