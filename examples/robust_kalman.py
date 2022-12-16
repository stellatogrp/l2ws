from ast import Return
import functools
import hydra
import cvxpy as cp
from jax import linear_transpose
import pandas as pd
from pandas import read_csv
# from l2ws.scs_problem import SCSinstance, scs_jax, ruiz_equilibrate
import numpy as np
import pdb
# from l2ws.launcher import Workspace
from scipy import sparse
import jax.numpy as jnp
from scipy.sparse import coo_matrix, bmat, csc_matrix
import jax.scipy as jsp
import time
import matplotlib.pyplot as plt
import os
import scs
import logging
from scipy import sparse
import yaml
from jax import jit, vmap
import cvxpy as cp

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
    }
)
log = logging.getLogger(__name__)


def simulate(T, gamma, dt, sigma, p):
    A, B, C = robust_kalman_setup(gamma, dt)

    # generate random input and noise vectors
    w = np.random.randn(2, T)
    v = np.random.randn(2, T)

    x = np.zeros((4, T + 1))
    x[:, 0] = [0, 0, 0, 0]
    y = np.zeros((2, T))

    # add outliers to v
    # np.random.seed(0)
    inds = np.random.rand(T) <= p
    v[:, inds] = sigma * np.random.randn(2, T)[:, inds]

    # simulate the system forward in time
    for t in range(T):
        y[:, t] = C.dot(x[:, t]) + v[:, t]
        x[:, t + 1] = A.dot(x[:, t]) + B.dot(w[:, t])

    x_true = x.copy()
    w_true = w.copy()
    return y, x_true, w_true, v


def test_kalman():
    T = 10
    gamma = 0.05
    dt = 0.05
    p = 20
    sigma = 20
    mu = 2
    rho = 2
    y, x_true, w_true, v = simulate(T, gamma, dt, sigma, p)

    # with huber
    x1, w1 = cvxpy_huber(T, y, gamma, dt, mu, rho)

    # with our canon, but cvxpy
    x2, w2 = cvxpy_manual(T, y, gamma, dt, mu, rho)
    pdb.set_trace()


def cvxpy_huber(T, y, gamma, dt, mu, rho):
    x = cp.Variable(shape=(4, T + 1))
    w = cp.Variable(shape=(2, T))
    v = cp.Variable(shape=(2, T))
    A, B, C = robust_kalman_setup(gamma, dt)

    obj = cp.sum_squares(w)
    obj += cp.sum([mu * cp.huber(cp.norm(v[:, t]), rho) for t in range(T)])
    obj = cp.Minimize(obj)

    constr = []
    for t in range(T):
        constr += [
            x[:, t + 1] == A * x[:, t] + B * w[:, t],
            y[:, t] == C * x[:, t] + v[:, t],
        ]

    cp.Problem(obj, constr).solve(verbose=True)

    x = np.array(x.value)
    w = np.array(w.value)
    return x, w

def cvxpy_manual(T, y, gamma, dt, mu, rho):
    x = cp.Variable(shape=(4, T + 1))
    w = cp.Variable(shape=(2, T))
    v = cp.Variable(shape=(2, T))
    s = cp.Variable(shape=(T))
    u = cp.Variable(shape=(T))
    z = cp.Variable(shape=(T))
    A, B, C = robust_kalman_setup(gamma, dt)

    obj = cp.sum_squares(w)
    obj += cp.sum([mu * (2 * u[t] + rho*z[t]**2) for t in range(T)])
    # obj += cp.sum([tau * cp.huber(cp.norm(v[:, t]), rho) for t in range(T)])
    obj = cp.Minimize(obj)

    constr = []
    for t in range(T):
        constr += [
            x[:, t + 1] == A @ x[:, t] + B @ w[:, t],
            y[:, t] == C @ x[:, t] + v[:, t],
            z <= rho,
            u >= 0,
            u + z == s,
            cp.norm(v[:,t]) <= s[t]
        ]

    cp.Problem(obj, constr).solve(verbose=True)

    x = np.array(x.value)
    w = np.array(w.value)
    return x, w



# @functools.partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8,))
def single_q(theta, m, n, T, nx, nu, state_box, control_box, A_dynamics):
    """
    the observations, y_0,...,y_{T-1} are the parameters that change
    there are 3 blocks of constraints
    1. x_{t+1}=Ax_t+Bw_t
    2. y_t=Cx_t+v_t
    3.
    """
    q = jnp.zeros(n + m)
    beq = jnp.zeros(T * nx)
    beq = beq.at[:nx].set(A_dynamics @ theta)
    b_upper = jnp.hstack([state_box * jnp.ones(T * nx), control_box * jnp.ones(T * nu)])
    b_lower = jnp.hstack([state_box * jnp.ones(T * nx), control_box * jnp.ones(T * nu)])
    b = jnp.hstack([beq, b_upper, b_lower])
    q = q.at[n:].set(b)
    return q


def get_q_mat_control_box_only(
    thetas, m, n, T, nx, nu, state_box, control_box, QB, vecc_gen
):
    N, nx = thetas.shape
    q_mat = control_box * jnp.ones((N, n + m))
    for i in range(N):
        c = QB @ thetas[i, :]
        b = control_box * jnp.ones(m)
        if state_box != "inf":
            start = 2 * T * nu
            rhs = vecc_gen @ thetas[i, :]
            b = b.at[start : start + T * nx].set(state_box - rhs)
            b = b.at[start + T * nx : start + 2 * T * nx].set(state_box + rhs)
        q_mat = q_mat.at[i, :n].set(c)
    return q_mat


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
    example = "robust_kalman"
    data_yaml_filename = f"{orig_cwd}/outputs/{example}/aggregate_outputs/{datetime}/data_setup_copied.yaml"

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    T, control_box = setup_cfg["T"], setup_cfg["control_box"]
    state_box = setup_cfg["state_box"]
    nx, nu = setup_cfg["nx"], setup_cfg["nu"]
    Q_val, QT_val = setup_cfg["Q_val"], setup_cfg["QT_val"]
    R_val = setup_cfg["R_val"]

    Ad, Bd = oscillating_masses_setup(nx, nu)

    static_dict = static_canon(
        T, nx, nu, state_box, control_box, Q_val, QT_val, R_val, Ad=Ad, Bd=Bd
    )
    A_sparse = static_dict["A_sparse"]
    m, n = A_sparse.shape

    get_q_single = functools.partial(
        single_q,
        m=m,
        n=n,
        T=T,
        nx=nx,
        nu=nu,
        state_box=state_box,
        control_box=control_box,
        A_dynamics=Ad,
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

    """
    create the dynamics depending on the system
    """
    Ad, Bd = robust_kalman_setup(cfg.nx, cfg.nu)

    """
    - canonicalize according to whether we have states or not
    - extract information dependent on the setup
    """
    log.info("creating static canonicalization...")
    t0 = time.time()
    out_dict = static_canon(
        cfg.T,
        cfg.nx,
        cfg.nu,
        cfg.state_box,
        cfg.control_box,
        cfg.Q_val,
        cfg.QT_val,
        cfg.R_val,
        Ad=Ad,
        Bd=Bd,
    )

    t1 = time.time()
    log.info(f"finished static canonicalization - took {t1-t0} seconds")

    M = out_dict["M"]
    algo_factor = out_dict["algo_factor"]
    cones_array = out_dict["cones_array"]
    A_sparse, P_sparse = out_dict["A_sparse"], out_dict["P_sparse"]

    """
    if with_states, b is updated
    if w/out states, c is updated
    """
    b, c = out_dict["b"], out_dict["c"]

    m, n = A_sparse.shape
    cones_dict = dict(z=int(cones_array[0]), l=int(cones_array[1]))

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
    tol = cfg.solve_acc
    solver = scs.SCS(data, cones_dict, eps_abs=tol, eps_rel=tol)
    solve_times = np.zeros(N)
    x_stars = jnp.zeros((N, n))
    y_stars = jnp.zeros((N, m))
    s_stars = jnp.zeros((N, m))
    q_mat = jnp.zeros((N, m + n))

    """
    initialize x_init over all the problems
    """
    # x_init_mat initialized uniformly between x_init_box*[-1,1]
    x_init_mat = cfg.x_init_box * (2 * np.random.rand(N, cfg.nx) - 1)

    scs_instances = []
    for i in range(N):
        infeasible = True
        while infeasible:
            log.info(f"solving problem number {i}")
            b[: cfg.nx] = Ad @ x_init_mat[i, :]

            # manual canon
            manual_canon_dict = {
                "P": P_sparse,
                "A": A_sparse,
                "b": b,
                "c": c,
                "cones": cones_dict,
            }
            scs_instance = SCSinstance(manual_canon_dict, solver, manual_canon=True)

            """
            check feasibility
            if infeasible, resample
            """
            if jnp.isnan(scs_instance.x_star[0]):
                x0 = cfg.x_init_box * (2 * np.random.rand(cfg.nx) - 1)
                x_init_mat[i, :] = x0
            else:
                infeasible = False

            scs_instances.append(scs_instance)
            x_stars = x_stars.at[i, :].set(scs_instance.x_star)
            y_stars = y_stars.at[i, :].set(scs_instance.y_star)
            s_stars = s_stars.at[i, :].set(scs_instance.s_star)
            q_mat = q_mat.at[i, :].set(scs_instance.q)
            solve_times[i] = scs_instance.solve_time

            ############ check with our jax implementation
            # P_jax = jnp.array(P_sparse.todense())
            # A_jax = jnp.array(A_sparse.todense())
            # c_jax, b_jax = jnp.array(c), jnp.array(b)
            # data = dict(P=P_jax, A=A_jax, b=b_jax, c=c_jax, cones=cones_dict)
            # # data['x'] = x_stars[i, :]
            # # data['y'] = y_stars[i, :]
            # x_jax, y_jax, s_jax = scs_jax(data, iters=1000)

            ############
            # qq = single_q(x_init_mat[0, :], m, n, cfg.T, cfg.nx, cfg.nu, cfg.state_box, cfg.control_box, Ad)
            # pdb.set_trace()

    # resave the data??
    # print('saving final data...', flush=True)
    log.info("saving final data...")
    t0 = time.time()
    jnp.savez(
        output_filename,
        thetas=x_init_mat,
        x_stars=x_stars,
        y_stars=y_stars,
    )
    # print(f"finished saving final data... took {save_time-t0}'", flush=True)
    save_time = time.time()
    log.info(f"finished saving final data... took {save_time-t0}'")

    # save plot of first 5 solutions
    for i in range(5):
        plt.plot(x_stars[i, :])
    plt.savefig("opt_solutions.pdf")

    # save plot of first 5 parameters
    for i in range(5):
        plt.plot(x_init_mat[i, :])
    plt.savefig("thetas.pdf")


def our_scs():
    # setup problem
    # T, nx, nu, control_box, x_init_box = 10, 200, 200, .1, .5
    T, nx, nu, control_box, x_init_box = 3, 10, 10, 0.1, 0.5
    np.random.seed(0)
    # np.eye(nx) + .02 * np.random.normal(size=(nx, nx))
    Ad = 0.05 * np.random.normal(size=(nx, nx))
    Bd = 1 * np.random.normal(size=(nx, nu))

    """
    states as variables
    """
    # canonicalize with states as variables
    t0 = time.time()
    out_dict = static_canon(T, nx, nu, control_box, Ad=Ad, Bd=Bd, state_box=100)
    t1 = time.time()
    print(f"canonicalized in {t1-t0} seconds")
    P_sparse, A_sparse = out_dict["P_sparse"], out_dict["A_sparse"]
    A_dynamics = out_dict["A_dynamics"]
    c, b = out_dict["c"], out_dict["b"]
    cones_array = out_dict["cones_array"]
    cones_dict = dict(z=int(cones_array[0]), l=int(cones_array[1]))

    # update x_init
    x_init = x_init_box * (2 * jnp.array(np.random.rand(nx)) - 1)
    b[:nx] = -A_dynamics @ x_init

    # input into our_scs
    P, A = P_sparse.todense(), A_sparse.todense()
    P_jax, A_jax = jnp.array(P), jnp.array(A)
    b_jax, c_jax = jnp.array(b), jnp.array(c)
    data = dict(P=P_jax, A=A_jax, b=b_jax, c=c_jax, cones=cones_dict)

    # solve with c scs
    data_c = dict(P=P_sparse, A=A_sparse, b=b, c=c)
    solver = scs.SCS(data_c, cones_dict, eps_abs=1e-5, eps_rel=1e-5, max_iters=10000)

    """
    only controls as inputs
    """
    # canonicalize with only controls as inputs
    t0 = time.time()
    out_dict = static_canon_control_box_only(T, nx, nu, control_box, Ad=Ad, Bd=Bd)
    t1 = time.time()
    print(f"canonicalized in {t1-t0} seconds")
    P_sparse, A_sparse = out_dict["P_sparse"], out_dict["A_sparse"]
    A_dynamics = out_dict["A_dynamics"]
    QB = out_dict["QB"]
    c, b = out_dict["c"], out_dict["b"]
    cones_array = out_dict["cones_array"]
    cones_dict = dict(z=int(cones_array[0]), l=int(cones_array[1]))

    # update x_init
    # x_init = x_init_box*(2*jnp.array(np.random.rand(nx))-1)
    c = np.array(QB @ x_init)  # * 2

    # input into our_scs
    P, A = P_sparse.todense(), A_sparse.todense()
    P_jax, A_jax = jnp.array(P), jnp.array(A)
    b_jax, c_jax = jnp.array(b), jnp.array(c)
    data = dict(P=P_jax, A=A_jax, b=b_jax, c=c_jax, cones=cones_dict)

    # solve with c scs
    pdb.set_trace()
    data_c = dict(P=P_sparse, A=A_sparse, b=b, c=c)
    solver_no_state = scs.SCS(
        data_c, cones_dict, eps_abs=1e-3, eps_rel=1e-8, max_iters=10000
    )

    # no learning
    print("about to solve with c code")
    t0 = time.time()
    m, n = A.shape
    sol = solver.solve()
    sol_no_state = solver_no_state.solve()
    t1 = time.time()
    print(f"solved with c code in {t1-t0} seconds")
    P_hat = ruiz_equilibrate(P)
    pdb.set_trace()
    print("true x sol", sol["x"][:5])
    print("true y sol", sol["y"][:5])

    # solve with our scs
    # data['x'], data['y'] = sol_no_state['x'], sol_no_state['y']
    x_jax, y_jax, s_jax = scs_jax(data, iters=500)
    print("our x sol", x_jax[:5])
    print("our y sol", y_jax[:5])
    pdb.set_trace()


def static_canon(T, mu, rho, dt):
    """
    variables
    (x_t, w_t, s_t, v_t,  u_t, z_t) \in (nx + nu + no + 3)
    (nx,  nu,  1,   no,   no, 1, 1)
    min \sum_{i=0}^{T-1} ||w_t||_2^2 + mu (u_t+z_t^2)^2
        s.t. x_{t+1} = Ax_t + Bw_t  t=0,...,T-1 (dyn)
             y_t = Cx_t + v_t       t=0,...,T-1 (obs)
             u_t + z_t = s_t        t=0,...,T-1 (aux)
             z_t <= 1               t=0,...,T-1 (z ineq)
             u_t >= 0               t=0,...,T-1 (u ineq)
             ||v_t||_2 <= s_t       t=0,...,T-1 (socp)
    (x_0, ..., x_{T-1})
    """
    # to make indexing easier
    single_len = nx + nu + no + 3
    nvars = single_len * T
    w_start = nx * T
    s_start = w_start + nu * T
    v_start = s_start + T
    u_start = v_start + no * T
    z_start = u_start + nu * T
    assert z_start + T == single_len * T

    # get A, B, C
    Ad, Bd, C = robust_kalman_setup(gamma, dt)

    # Quadratic objective
    P = np.zeros((single_len, single_len))
    P[nx : nx + nu, nx : nx + nu] = np.eye(nu)
    P[-1, -1] = mu
    P_sparse = sparse.kron(sparse.eye(T - 1), P)

    # Linear objective
    c = np.zeros(single_len * T)
    c[-2 * T : -T] = 2 * mu

    # dyn constraints
    Ax = sparse.kron(sparse.eye(T), -sparse.eye(nx)) + sparse.kron(
        sparse.eye(T, k=-1), Ad
    )
    Bw = sparse.kron(sparse.eye(T), Bd)
    A_dyn = np.zeros((T * nx, nvars))
    A_dyn[:, :w_start] = Ax
    A_dyn[:, w_start:s_start] = Bw
    b_dyn = np.zeros(T * nx)

    # obs constraints
    Cx = sparse.kron(sparse.eye(T), C)

    Iv = sparse.kron(sparse.eye(T), sparse.eye(T))
    A_obs = np.zeros((T * no, nvars))
    A_obs[:, :w_start] = Cx
    A_obs[:, v_start:u_start] = Iv
    # b_obs will be updated by the parameter stack(y_1, ..., y_T)
    b_obs = np.zeros(T * no)

    # aux constraints
    n_aux = T
    A_aux = np.zeros((n_aux, nvars))
    A_aux[:, s_start:v_start] = -I
    A_aux[:, u_start:z_start] = I
    A_aux[:, z_start:] = I
    b_aux = np.zeros(n_aux)

    # z_ineq constraints

    # u_ineq constraints

    # stack A
    A_sparse = sparse.vstack([A_dyn, A_obs, A_aux])

    # get b
    b = np.hstack([b_dyn, b_obs, b_aux])

    cones = dict(z=T * (1 + nx + no), l=2 * (T * nx + T * nu))
    cones_array = jnp.array([cones["z"], cones["l"], cones["s"]])

    # create the matrix M
    m, n = A_sparse.shape
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
        cones_array=cones_array,
        A_sparse=A_sparse,
        P_sparse=P_sparse,
        b=b,
        c=c,
        A_dynamics=Ad,
    )

    return out_dict


def robust_kalman_setup(gamma, dt):
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    C = np.zeros((2, 4))

    A[0, 0] = 1
    A[1, 1] = 1
    A[0, 2] = (1 - gamma * dt / 2) * dt
    A[1, 3] = (1 - gamma * dt / 2) * dt
    A[2, 2] = 1 - gamma * dt
    A[3, 3] = 1 - gamma * dt

    B[0, 0] = dt**2 / 2
    B[1, 1] = dt**2 / 2
    B[2, 0] = dt
    B[3, 1] = dt

    C[0, 0] = 1
    C[1, 1] = 1

    return A, B, C


"""
pipeline is as follows

static_canon --> given (nx, nu, T) get (P, c, A), also M and factor
-- DONT store in memory

setup data,
-- input from cfg: (N, nx, nu, T)
-- output: (x_stars, y_stars, s_stars, thetas, q_mat??)

q_mat ?? we can directly get it from thetas in this case

possibly for each example, pass in get_q_mat function
"""


if __name__ == "__main__":
    test_kalman()
