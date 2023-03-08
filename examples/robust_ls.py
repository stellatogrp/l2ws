from functools import partial
import hydra
import cvxpy as cp
from l2ws.scs_problem import SCSinstance, scs_jax
import numpy as np
from l2ws.launcher import Workspace
import jax.numpy as jnp
from scipy.sparse import csc_matrix
import jax.scipy as jsp
import time
import matplotlib.pyplot as plt
import os
import scs
import logging
import yaml
from jax import vmap
import pandas as pd


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
    }
)
log = logging.getLogger(__name__)


def cvxpy_manual(A, b, rho):
    m, n = A.shape
    x = cp.Variable(shape=(n))
    u = cp.Variable()
    v = cp.Variable()

    obj = u + rho * v

    constr = [x >= 0, cp.norm(x) <= v, cp.norm(A @ x - b) <= u]

    prob = cp.Problem(cp.Minimize(obj), constr).solve(verbose=True)
    prob.solve(verbose=True)

    x = np.array(x.value)
    u = np.array(u.value)
    v = np.array(v.value)
    return x, u, v


# @functools.partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8,))
# def single_q(theta, m, n, T, nx, nu, state_box, control_box, A_dynamics):
def single_q(theta, rho, m_orig, n_orig):
    # note: m, n are the sizes of the constraint matrix in the SOCP
    # theta is the vector b
    m = 2 * m_orig + n_orig + 2
    n = n_orig + 2

    # c
    c = jnp.zeros(n)
    c = c.at[-1].set(rho)
    c = c.at[-2].set(1)

    # b
    b = jnp.zeros(m)
    b = b.at[n_orig: n_orig + m_orig].set(theta)

    # q
    m = b.size
    q = jnp.zeros(m + n)
    q = q.at[:n].set(c)
    q = q.at[n:].set(b)

    return q


def run(run_cfg):
    """
    retrieve data for this config
    theta is all of the following
    theta = (ret, pen_risk, pen_hold, pen_trade, w0)

    Sigma is constant

     just need (theta, factor, u_star), Pi
    """
    # todo: retrieve data and put into a nice form - OR - just save to nice form

    """
    create workspace
    needs to know the following somehow -- from the run_cfg
    1. nn cfg
    2. (theta, factor, u_star)_i=1^N
    3. Pi

    2. and 3. are stored in data files and the run_cfg holds the location

    it will create the l2a_model
    """
    datetime = run_cfg.data.datetime
    orig_cwd = hydra.utils.get_original_cwd()
    example = "robust_ls"
    folder = f"{orig_cwd}/outputs/{example}/aggregate_outputs/{datetime}"
    data_yaml_filename = f"{folder}/data_setup_copied.yaml"

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    # T, control_box = setup_cfg["T"], setup_cfg["control_box"]
    # state_box = setup_cfg["state_box"]
    # nx, nu = setup_cfg["nx"], setup_cfg["nu"]
    # Q_val, QT_val = setup_cfg["Q_val"], setup_cfg["QT_val"]
    # R_val = setup_cfg["R_val"]

    # Ad, Bd = oscillating_masses_setup(nx, nu)

    # set the seed
    np.random.seed(setup_cfg['seed'])
    m_orig, n_orig = setup_cfg['m_orig'], setup_cfg['n_orig']
    rho = setup_cfg['rho']

    # create the nominal matrix A
    A = (np.random.rand(m_orig, n_orig) * 2) - 1

    static_dict = static_canon(A, rho)

    get_q_single = partial(single_q,
                           rho=setup_cfg['rho'],
                           m_orig=setup_cfg['m_orig'],
                           n_orig=setup_cfg['n_orig'],
                           )

    get_q = vmap(get_q_single, in_axes=0, out_axes=0)

    """
    static_flag = True
    means that the matrices don't change across problems
    we only need to factor once
    """
    static_flag = True
    workspace = Workspace(run_cfg, static_flag, static_dict, example, get_q)

    """
    run the workspace
    """
    workspace.run()


def setup_probs(setup_cfg):
    print("entered robust kalman setup", flush=True)
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    m_orig, n_orig = cfg.m_orig, cfg.n_orig

    np.random.seed(cfg.seed)
    # A = np.random.normal(size=(m_orig, n_orig))
    A = (np.random.rand(m_orig, n_orig) * 2) - 1

    log.info("creating static canonicalization...")
    t0 = time.time()
    out_dict = static_canon(A, cfg.rho)

    t1 = time.time()
    log.info(f"finished static canonicalization - took {t1-t0} seconds")

    # M = out_dict["M"]
    # algo_factor = out_dict["algo_factor"]
    # cones_array = out_dict["cones_array"]
    cones_dict = out_dict["cones_dict"]
    A_sparse, P_sparse = out_dict["A_sparse"], out_dict["P_sparse"]

    """
    if with_states, b is updated
    if w/out states, c is updated
    """
    b, c = out_dict["b"], out_dict["c"]

    m, n = A_sparse.shape
    # cones_dict = dict(z=int(cones_array[0]), l=int(cones_array[1]))

    """
    save output to output_filename
    """
    # save to outputs/mm-dd-ss/... file
    if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
        slurm_idx = os.environ["SLURM_ARRAY_TASK_ID"]
        output_filename = f"{os.getcwd()}/data_setup_slurm_{slurm_idx}"
    else:
        output_filename = f"{os.getcwd()}/data_setup_slurm"
    """
    create scs solver object
    we can cache the factorization if we do it like this
    """

    data = dict(P=P_sparse, A=A_sparse, b=b, c=c)
    tol_abs = cfg.solve_acc_abs
    tol_rel = cfg.solve_acc_rel
    solver = scs.SCS(data, cones_dict, eps_abs=tol_abs, eps_rel=tol_rel)
    solve_times = np.zeros(N)
    x_stars = jnp.zeros((N, n))
    y_stars = jnp.zeros((N, m))
    s_stars = jnp.zeros((N, m))
    q_mat = jnp.zeros((N, m + n))

    """
    sample theta for each problem
    """
    thetas_np = (2 * np.random.rand(N, m_orig) - 1) * cfg.b_range + cfg.b_nominal
    thetas = jnp.array(thetas_np)

    batch_q = vmap(single_q, in_axes=(0, None, None, None), out_axes=(0))

    q_mat = batch_q(thetas, cfg.rho, cfg.m_orig, cfg.n_orig)

    scs_instances = []

    for i in range(N):
        log.info(f"solving problem number {i}")
        # print(f"solving problem number {i}")

        # update
        b = np.array(q_mat[i, n:])
        c = np.array(q_mat[i, :n])

        # manual canon
        manual_canon_dict = {
            "P": P_sparse,
            "A": A_sparse,
            "b": b,
            "c": c,
            "cones": cones_dict,
        }
        scs_instance = SCSinstance(manual_canon_dict, solver, manual_canon=True)

        scs_instances.append(scs_instance)
        x_stars = x_stars.at[i, :].set(scs_instance.x_star)
        y_stars = y_stars.at[i, :].set(scs_instance.y_star)
        s_stars = s_stars.at[i, :].set(scs_instance.s_star)
        q_mat = q_mat.at[i, :].set(scs_instance.q)
        solve_times[i] = scs_instance.solve_time

        # check with our jax implementation
        P_jax = jnp.array(P_sparse.todense())
        A_jax = jnp.array(A_sparse.todense())
        c_jax, b_jax = jnp.array(c), jnp.array(b)
        data = dict(P=P_jax, A=A_jax, b=b_jax, c=c_jax, cones=cones_dict)
        # disturbance_x = np.random.normal(size=(n))
        # disturbance_x = disturbance_x / np.linalg.norm(disturbance_x) * 1e-2
        # disturbance_y = np.random.normal(size=(m))
        # disturbance_y = disturbance_y / np.linalg.norm(disturbance_y) * 1e-2
        # data['x'] = x_stars[i, :] + disturbance_x
        # data['y'] = y_stars[i, :] + disturbance_y
        x_jax, y_jax, s_jax = scs_jax(data, iters=1000)

        ############

    # resave the data??
    # print('saving final data...', flush=True)
    log.info("saving final data...")
    t0 = time.time()
    jnp.savez(
        output_filename,
        thetas=thetas,
        x_stars=x_stars,
        y_stars=y_stars
    )

    # save solve times
    df_solve_times = pd.DataFrame(solve_times, columns=['solve_times'])
    df_solve_times.to_csv('solve_times.csv')

    # print(f"finished saving final data... took {save_time-t0}'", flush=True)
    save_time = time.time()
    log.info(f"finished saving final data... took {save_time-t0}'")

    # save plot of first 5 solutions
    for i in range(5):
        plt.plot(x_stars[i, :])
    plt.savefig("x_stars.pdf")
    plt.clf()

    # save plot of first 5 solutions - just x
    for i in range(5):
        plt.plot(x_stars[i, :-2])
    plt.savefig("x_stars_just_x.pdf")
    plt.clf()

    # save plot of first 5 solutions - non-zeros
    for i in range(5):
        plt.plot(x_stars[i, :-2] >= 0.0001)
    plt.savefig("x_stars_zero_one.pdf")
    plt.clf()

    # correlation matrix
    corrcoef = np.corrcoef(x_stars[:, :-2] >= 0.0001)
    print('corrcoef', corrcoef)
    plt.imshow(corrcoef)
    plt.savefig("corrcoef_zero_one.pdf")
    plt.clf()

    for i in range(5):
        plt.plot(y_stars[i, :])
    plt.savefig("y_stars.pdf")
    plt.clf()

    # save plot of first 5 parameters
    for i in range(5):
        plt.plot(thetas[i, :])
    plt.savefig("thetas.pdf")
    plt.clf()


def static_canon(A, rho):
    """
    Let A have shape (m_orig, n_orig)
    min_{x,u,v} u + rho v
                s.t. x >= 0
                     ||Ax-b||_2 <= u
                     ||x||_2 <= v

    min_{x,u,v} u + rho v
                s.t. -x + s_1 == 0              (n)
                     -(u, Ax) + s_2 == (-b, 0)  (m, 1)
                     -(v, x) + s_3 == 0         (n, 1)
                     s_1 in R^n+
                     s_2 in SOC(m, 1)
                     s_3 in SOC(n, 1)

    in total:
    m = 2 * n_orig + m_orig + 2 constraints
    n = n_orig + 2 vars

    Assume that A is fixed from problem to problem

    vars = (x, u, v)
    c = (0, 1, rho)
    """
    m_orig, n_orig = A.shape
    m, n = 2 * n_orig + m_orig + 2, n_orig + 2
    A_dense = np.zeros((m, n))
    b = np.zeros(m)

    # constraint 1
    A_dense[:n_orig, :n_orig] = -np.eye(n_orig)

    # constraint 2
    A_dense[n_orig, n_orig] = -1
    A_dense[n_orig + 1:n_orig + m_orig + 1, :n_orig] = -A

    b[n_orig:m_orig + n_orig] = 0  # fill in for b when theta enters --
    # here we can put anything since it will change

    # constraint 3
    A_dense[n_orig + m_orig + 1, n_orig + 1] = -1
    A_dense[n_orig + m_orig + 2:, :n_orig] = -np.eye(n_orig)

    # create sparse matrix
    A_sparse = csc_matrix(A_dense)

    # cones
    q_array = [m_orig + 1, n_orig + 1]
    cones = dict(z=0, l=n_orig, q=q_array)
    cones_array = jnp.array([cones["z"], cones["l"]])
    cones_array = jnp.concatenate([cones_array, jnp.array(cones["q"])])

    # Quadratic objective
    P = np.zeros((n, n))
    P_sparse = csc_matrix(P)

    # Linear objective
    c = np.zeros(n)
    c[n_orig], c[n_orig + 1] = 1, rho

    # create the matrix M
    M = jnp.zeros((n + m, n + m))
    P = P_sparse.todense()
    A = A_sparse.todense()
    P_jax = jnp.array(P)
    A_jax = jnp.array(A)
    M = M.at[:n, :n].set(P_jax)
    M = M.at[:n, n:].set(A_jax.T)
    M = M.at[n:, :n].set(-A_jax)

    # factor for DR splitting
    algo_factor = jsp.linalg.lu_factor(M + jnp.eye(n + m))

    A_sparse = csc_matrix(A)
    P_sparse = csc_matrix(P)

    out_dict = dict(
        M=M,
        algo_factor=algo_factor,
        cones_dict=cones,
        cones_array=cones_array,
        A_sparse=A_sparse,
        P_sparse=P_sparse,
        b=b,
        c=c
    )
    return out_dict


def random_robust_ls(m_orig, n_orig, rho, b_center, b_range, seed=42):
    """
    given dimensions, returns a random robust least squares problem
    """
    A = (np.random.rand(m_orig, n_orig) * 2) - 1
    out = static_canon(A, rho)
    c, b = out['c'], out['b']
    P_sparse, A_sparse = out['P_sparse'], out['A_sparse']
    P, A = jnp.array(P_sparse.todense()), jnp.array(A_sparse.todense())
    cones = out['cones_dict']

    b_rand_np = (2 * np.random.rand(m_orig) - 1) * b_range + b_center
    b[n_orig:m_orig + n_orig] = np.array(b_rand_np)
    return P, A, jnp.array(c), jnp.array(b), cones
