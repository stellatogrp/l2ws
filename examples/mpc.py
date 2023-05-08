import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from scipy.linalg import solve_discrete_are
from examples.osc_mass import static_canon_osqp
import cvxpy as cp
import yaml
from l2ws.launcher import Workspace
import os
from examples.solve_script import 


def run(run_cfg):
    example = "mpc"
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
    A = jnp.array(np.random.normal(size=(m_orig, n_orig)))
    evals, evecs = jnp.linalg.eigh(A.T @ A)
    ista_step =  1 / evals.max()
    lambd = setup_cfg['lambd']

    static_dict = dict(A=A, lambd=lambd, ista_step=ista_step)

    # we directly save q now
    static_flag = True
    algo = 'osqp'
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example)

    # run the workspace
    workspace.run()


def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    np.random.seed(setup_cfg['seed'])
    m_orig, n_orig = setup_cfg['m_orig'], setup_cfg['n_orig']
    A = jnp.array(np.random.normal(size=(m_orig, n_orig)))
    evals, evecs = jnp.linalg.eigh(A.T @ A)
    ista_step = 1 / evals.max()
    lambd = setup_cfg['lambd']

    np.random.seed(cfg.seed)

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    # P, A, cones, q_mat, theta_mat_jax, A_tensor = multiple_random_sparse_pca(
    #     n_orig, cfg.k, cfg.r, N, factor=False)
    # P_sparse, A_sparse = csc_matrix(P), csc_matrix(A)
    b_mat = generate_b_mat(A, N, p=.1)
    m, n = A.shape

    # create scs solver object
    #    we can cache the factorization if we do it like this
    # b_np, c_np = np.array(q_mat[0, n:]), np.array(q_mat[0, :n])
    # data = dict(P=P_sparse, A=A_sparse, b=b_np, c=c_np)
    # tol_abs = cfg.solve_acc_abs
    # tol_rel = cfg.solve_acc_rel
    # solver = scs.SCS(data, cones, normalize=False, alpha=1, scale=1,
    #                  rho_x=1, adaptive_scale=False, eps_abs=tol_abs, eps_rel=tol_rel)

    osqp_setup_script(b_mat, A, lambd, output_filename)


def generate_static_prob_data(nx, nu, seed):
    np.random.seed(seed)
    x_bar = 1 + np.random.rand(nx)
    u_bar = .5 * np.random.rand(nu)

    dA = .1 * np.random.normal(size=(nx, nx))
    orig_Ad = np.eye(nx) + dA

    # normalize Ad so eigenvalues are less than 1
    orig_evals, evecs = np.linalg.eig(orig_Ad)
    # import pdb
    # pdb.set_trace()
    max_norm = np.max(np.abs(orig_evals))
    if max_norm >= 1:
        Ad = orig_Ad / max_norm
    else:
        Ad = orig_Ad

    # evals = np.clip(orig_evals, a_min=-np.inf, a_max=1)
    # Ad = evecs @ np.diag(evals) @ evecs.T

    # new_evals, new_evecs = np.linalg.eigh(orig_Ad)

    Bd = np.random.normal(size=(nx, nu))

    # create cost matrices
    R = .1 * np.eye(nu)

    q_vec = np.random.rand(nx) * 10
    p = 0.7 #1.0
    q_vec_mask = np.random.choice([0, 1], size=(nx), p=[1-p, p], replace=True)
    q_vec = np.multiply(q_vec, q_vec_mask)
    Q = np.diag(q_vec)

    QT = solve_discrete_are(Ad, Bd, Q, R)
    # QT = np.eye(nx)

    return Ad, Bd, Q, QT, R, x_bar, u_bar


def multiple_random_mpc_osqp(N, 
                             T=10, 
                             x_init_factor=.3, 
                             nx=20, 
                             nu=10,
                             sigma=1, 
                             rho=1,
                             Ad=None,
                             Bd=None,
                             seed=42):
    Ad, Bd, Q, QT, R, x_bar, u_bar = generate_static_prob_data(nx, nu, seed)
    static_dict = static_canon_osqp(T, nx, nu, x_bar, u_bar, Q,
                                    QT, R, Ad=Ad, Bd=Bd)
    P, A = static_dict['P'], static_dict['A']
    c, l, u = static_dict['c'], static_dict['l'], static_dict['u']
    m, n = A.shape
    Ad = static_dict['A_dynamics']
    cones = static_dict['cones']

    q_mat = jnp.zeros((N, n + 2 * m))
    q_mat = q_mat.at[:, :n].set(c)

    # factor
    rho_vec = jnp.ones(m)
    rho_vec = rho_vec.at[l == u].set(1000)

    # M = P + sigma * jnp.eye(n) + rho * A.T @ A
    M = P + sigma * jnp.eye(n) + A.T @ jnp.diag(rho_vec) @ A
    factor = jsp.linalg.lu_factor(M)

    # x_init is theta
    # x_init_mat = jnp.array(x_init_box * (2 * np.random.rand(N, nx) - 1))
    x_init_mat = x_init_factor * jnp.array(x_bar * (2 * np.random.rand(N, nx) - 1))

    for i in range(N):
        # generate new rhs of first block constraint
        l = l.at[:nx].set(Ad @ x_init_mat[i, :])
        u = u.at[:nx].set(Ad @ x_init_mat[i, :])

        q_osqp = jnp.concatenate([c, l, u])
        q_mat = q_mat.at[i, :].set(q_osqp)
    theta_mat = x_init_mat
    # return factor, P, A, q_mat, theta_mat

    return factor, P, A, q_mat, theta_mat, x_bar, Ad, rho_vec

def solve_many_probs_cvxpy(P, A, q_mat):
    """
    solves many QPs where each problem has a different b vector
    """
    P = cp.atoms.affine.wraps.psd_wrap(P)
    m, n = A.shape
    N = q_mat.shape[0]
    x, w = cp.Variable(n), cp.Variable(m)
    c_param, l_param, u_param = cp.Parameter(n), cp.Parameter(m), cp.Parameter(m)
    constraints = [A @ x == w, l_param <= w, w <= u_param]
    # import pdb
    # pdb.set_trace()
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, P) + c_param @ x), constraints)
    # prob = cp.Problem(cp.Minimize(.5 * cp.sum_squares(np.array(A) @ z - b_param) + lambd * cp.tv(z)))
    z_stars = jnp.zeros((N, m + n))
    objvals = jnp.zeros((N))
    for i in range(N):
        c_param.value = np.array(q_mat[i, :n])
        l_param.value = np.array(q_mat[i, n:n + m])
        u_param.value = np.array(q_mat[i, n + m:])
        prob.solve(verbose=False)
        objvals = objvals.at[i].set(prob.value)

        # import pdb
        # pdb.set_trace()
        x_star = jnp.array(x.value)
        w_star = jnp.array(w.value)
        y_star = jnp.array(constraints[0].dual_value)
        # z_star = jnp.concatenate([x_star, w_star, y_star])
        z_star = jnp.concatenate([x_star, y_star])
        z_stars = z_stars.at[i, :].set(z_star)
    print('finished solving cvxpy problems')
    return z_stars, objvals


def solve_multiple_trajectories(T, num_traj, x_bar, x_init_factor, Ad, P, A, q):
    m, n = A.shape
    nx = Ad.shape[0]
    q_mat = jnp.zeros((T * num_traj, n + 2 * m))
    first_x_inits = x_init_factor * jnp.array(x_bar * (2 * np.random.rand(num_traj, nx) - 1))
    theta_mat_list = []
    z_stars_list = []
    q_mat_list = []
    for i in range(num_traj):
        first_x_init = first_x_inits[i, :]
        theta_mat_curr, z_stars_curr, q_mat_curr = solve_trajectory(first_x_init, P, A, q, T, Ad)
        theta_mat_list.append(theta_mat_curr)
        z_stars_list.append(z_stars_curr)
        q_mat_list.append(q_mat_curr)
    theta_mat = jnp.vstack(theta_mat_list)
    z_stars = jnp.vstack(z_stars_list)
    q_mat = jnp.vstack(q_mat_list)
    # print('z_stars[0,:]', z_stars[0, :10])
    # print('z_stars[1,:]', z_stars[1, :10])
    # import pdb
    # pdb.set_trace()
    return theta_mat, z_stars, q_mat


def solve_trajectory(first_x_init, P, A, q, T, Ad):
    """
    given the system and a first x_init, we model the MPC paradigm

    solve the problem with first_x_init and implement the first control to get the second state
        that is the new x_init for the next problem

    returns
    1. theta_mat -- the initial states
    2. q_mat -- the problem data (could also be reverse engineered from theta_mat)
    3. z_stars -- the optimal solutions
    """
    nx = Ad.shape[0]

    # setup cvxpy problem
    P = cp.atoms.affine.wraps.psd_wrap(P)
    m, n = A.shape
    x, w = cp.Variable(n), cp.Variable(m)
    c_param, l_param, u_param = cp.Parameter(n), cp.Parameter(m), cp.Parameter(m)
    constraints = [A @ x == w, l_param <= w, w <= u_param]
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, P) + c_param @ x), constraints)

    c_param.value = np.array(q[:n])

    z_stars = jnp.zeros((T, m + n))
    theta_mat = jnp.zeros((T, nx))
    q_mat = jnp.zeros((T, n + 2 * m))
    q_mat = q_mat.at[:, :].set(q)

    # set the first x_init
    x_init = first_x_init

    for i in range(T):
        l = q[n:n + m]
        u = q[n + m:]
        theta_mat = theta_mat.at[i, :].set(x_init)
        Ad_x_init = Ad @ x_init
        l = l.at[:nx].set(Ad_x_init)
        u = u.at[:nx].set(Ad_x_init)
        l_param.value = np.array(l)
        u_param.value = np.array(u)
        prob.solve(verbose=False)

        x_star = jnp.array(x.value)
        w_star = jnp.array(w.value)
        y_star = jnp.array(constraints[0].dual_value)
        z_star = jnp.concatenate([x_star, y_star])
        # print('z_star', z_star[:20])
        z_stars = z_stars.at[i, :].set(z_star)

        q_mat = q_mat.at[i, n:n + m].set(l)
        q_mat = q_mat.at[i, n + m:].set(u)

        # set the next x_init
        x_init = -x_star[nx:2 * nx]
    return theta_mat, z_stars, q_mat
