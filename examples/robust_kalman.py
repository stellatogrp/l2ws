from functools import partial
import hydra
import cvxpy as cp
from l2ws.scs_problem import SCSinstance, scs_jax
import numpy as np
import pdb
from l2ws.launcher import Workspace
from scipy import sparse
import jax.numpy as jnp
from scipy.sparse import csc_matrix
import jax.scipy as jsp
import time
import matplotlib.pyplot as plt
import os
import scs
import logging
import yaml
from jax import vmap, jit
import pandas as pd
import matplotlib.colors as mc
import colorsys


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
    # x[:, 0] = [0, 0, 0, 0]
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


def sample_theta(N, T, sigma, p, gamma, dt, w_noise_var, y_noise_var, B_const=1):
    # generate random input and noise vectors
    w = w_noise_var * np.random.randn(N, 2, T)
    v = y_noise_var * np.random.randn(N, 2, T)
    # w = np.random.randn(scale=noise_var, size=(N, 2, T))
    # v = np.random.randn(scale=noise_var, size=(N, 2, T))

    # x = np.zeros((4, T + 1))
    # x[:, 0] = [0, 0, 0, 0]
    # y = np.zeros((2, T))

    # add outliers to v
    # np.random.seed(0)

    for j in range(N):
        inds = np.random.rand(T) <= p
        outlier_v = sigma * np.random.randn(2, T)[:, inds]
        # pdb.set_trace()

        # weird indexing to avoid transpose
        v[j:j+1, :, inds] = outlier_v

    # w_flat = np.ravel(w)
    # v_flat = np.ravel(v)
    # theta = np.concatenate([w_flat, v_flat])
    simulate_fwd_batch = vmap(simulate_fwd, in_axes=(0, 0, None, None, None, None), out_axes=(0, 0))

    # y_mat = simulate_fwd(w, v, T, gamma, dt)
    y_mat, x_trues = simulate_fwd_batch(w, v, T, gamma, dt, B_const)

    # get the rotation angle
    # y_mat has shape (N, 2, T)
    find_rotation_angle_vmap = vmap(find_rotation_angle, in_axes=(0,), out_axes=(0))

    angles = find_rotation_angle_vmap(y_mat[:, :, -1])

    # rotation_vmap = vmap(rotate_vector, in_axes=(0, 0), out_axes=(0))
    clockwise = True
    y_mat_rotated = rotation_vmap(y_mat, angles, clockwise)

    thetas = jnp.zeros((N, 2*T))
    for i in range(N):
        theta = jnp.ravel(y_mat_rotated[i, :, :].T)
        thetas = thetas.at[i, :].set(theta)

    w_trues = w
    # pdb.set_trace()
    state_pos = jnp.array([0, 1])
    x_states = x_trues[:, state_pos, :]
    x_trues_rotated = rotation_vmap(x_states, angles, clockwise)
    w_trues_rotated = rotation_vmap(w_trues, angles, clockwise)
    # pdb.set_trace()

    return thetas, y_mat, x_trues, w_trues, y_mat_rotated, x_trues_rotated, w_trues_rotated, angles


def rotate_vector(position, angle, clockwise):
    if clockwise:
        angle = -angle
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    rotation_matrix = jnp.array([[cos, -sin], [sin, cos]])
    new_position = rotation_matrix @ position
    return new_position


rotation_vmap = vmap(rotate_vector, in_axes=(0, 0, None), out_axes=(0))
rotation_single = vmap(rotate_vector, in_axes=(0, None, None), out_axes=(0))


def find_rotation_angle(y_T):
    """
    y_mat has shape (2, T)
    give (y_1, ..., y_T) where each entry is in R^2,
    returns the angle
    """
    # # only need the last entry, y_T
    # pos_angle = jnp.arctan(y_T[1] / y_T[0])

    # # if y_T[0] is positive, this is correct
    # # otherwise, angle is
    # if
    # angle = np.pi - pos_angle
    # return angle
    return jnp.arctan2(y_T[1], y_T[0])


def test_kalman():
    nx, no, nu = 4, 2, 2
    single_length = nx + no + nu + 3

    T = 30
    nvars = single_length * T
    gamma = 0.05
    dt = 0.5  # 0.05
    p = .2
    sigma = 20
    mu = 1
    rho = 1
    # y, x_true, w_true, v = simulate(T, gamma, dt, sigma, p)

    # sample to get (w, v) -- returns flat theta vector theta = (flat(w), flat(v))
    # theta_np = sample_theta(T, sigma, p)
    theta_np, x_trues, w_trues = sample_theta(T, sigma, p, gamma, dt)
    theta = jnp.array(theta_np)

    # get (x_true, w_true)
    x_true, w_true = get_x_w_true(theta, T, gamma, dt)

    # get y
    q = single_q(theta, mu, rho, T, gamma, dt)

    # update to input y
    c_np, b_np = np.array(q[:nvars]), np.array(q[nvars:])
    y = b_np[(T-1)*nx:T*(nx+no)-nx]
    # yy= y.copy()
    y_mat = np.reshape(y, (T, no))
    y_mat = y_mat.T

    # with huber
    x1, w1, v1 = cvxpy_huber(T, y_mat, gamma, dt, mu, rho)

    # plotting
    time_limit = dt * T
    ts, delt = np.linspace(0, time_limit, T, endpoint=True, retstep=True)
    # plot_state(ts,(x_true, w_true),(x1, w1))
    # plot_positions([x_true, y_mat], ['True', 'Noisy'])
    # plot_positions([x_true, x1], ['True', 'KF recovery'])

    # replace huber with socp, but cvxpy
    # x2, w2, v2 = cvxpy_manual(T, y_mat, gamma, dt, mu, rho)

    """
    scs with our canon
    """
    # canon
    out = static_canon(T, gamma, dt, mu, rho)
    cones_dict = out["cones_dict"]

    # solve with scs
    tol = 1e-8
    data = dict(P=out["P_sparse"], A=out["A_sparse"], c=c_np, b=b_np)

    solver = scs.SCS(data, cones_dict, eps_abs=tol, eps_rel=tol)
    sol = solver.solve()
    x = sol["x"]
    print('x', x)

    # x3 = x[: T * nx]
    # w3 = x[T * nx: T * (nx + nu)]
    # s3 = x[T * (nx + nu): T * (nx + nu + 1)]
    # v3 = x[T * (nx + nu + 1):-2*T]
    # u3 = x[-T*2:-T]
    # z3 = x[-T:]

    # (x_t, w_t, s_t, v_t,  u_t, z_t)

    # check with our scs
    P_jax = jnp.array(out["P_sparse"].todense())
    A_jax = jnp.array(out["A_sparse"].todense())
    c_jax = jnp.array(c_np)
    b_jax = jnp.array(b_np)

    cones_jax = out["cones_dict"]
    data = dict(P=P_jax, A=A_jax, c=c_jax, b=b_jax, cones=cones_jax)
    # data['x'], data['y'] = sol['x'], sol['y']

    xp, yd, sp = scs_jax(data, iters=1000)
    # x4 = xp[: T * nx]
    # w4 = xp[T * nx: T * (nx + nu)]
    # s4 = xp[T * (nx + nu): T * (nx + nu + 1)]
    # v4 = xp[T * (nx + nu + 1):-2*T]
    # u4 = xp[-T*2:-T]
    # z4 = xp[-T:]
    pdb.set_trace()


def plot_state(t, actual, estimated=None, filename=None):
    '''
    plot position, speed, and acceleration in the x and y coordinates for
    the actual data, and optionally for the estimated data
    '''
    trajectories = [actual]
    if estimated is not None:
        trajectories.append(estimated)

    fig, ax = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(8, 8))
    for x, w in trajectories:
        ax[0, 0].plot(t, x[0, :-1])
        ax[0, 1].plot(t, x[1, :-1])
        ax[1, 0].plot(t, x[2, :-1])
        ax[1, 1].plot(t, x[3, :-1])
        ax[2, 0].plot(t, w[0, :])
        ax[2, 1].plot(t, w[1, :])

    ax[0, 0].set_ylabel('x position')
    ax[1, 0].set_ylabel('x velocity')
    ax[2, 0].set_ylabel('x input')

    ax[0, 1].set_ylabel('y position')
    ax[1, 1].set_ylabel('y velocity')
    ax[2, 1].set_ylabel('y input')

    ax[0, 1].yaxis.tick_right()
    ax[1, 1].yaxis.tick_right()
    ax[2, 1].yaxis.tick_right()

    ax[0, 1].yaxis.set_label_position("right")
    ax[1, 1].yaxis.set_label_position("right")
    ax[2, 1].yaxis.set_label_position("right")

    ax[2, 0].set_xlabel('time')
    ax[2, 1].set_xlabel('time')
    if filename:
        fig.savefig(filename, bbox_inches='tight')
    else:
        plt.show()


def plot_positions(traj, labels, axis=None, filename=None):
    '''
    show point clouds for true, observed, and recovered positions
    '''
    # matplotlib.rcParams.update({'font.size': 14})
    n = len(traj)

    fig, ax = plt.subplots(1, n, sharex=True, sharey=True, figsize=(12, 5))
    if n == 1:
        ax = [ax]

    for i, x in enumerate(traj):
        ax[i].plot(x[0, :], x[1, :], 'ro', alpha=.1)
        ax[i].set_title(labels[i])
        if axis:
            ax[i].axis(axis)

    if filename:
        fig.savefig(filename, bbox_inches='tight')
    else:
        plt.show()


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    if color in mc.cnames.keys():
        c = mc.cnames[color]
    else:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_positions_overlay(traj, labels, num_dots=2, axis=None, filename=None):
    '''
    show point clouds for true, observed, and recovered positions

    the first num_dots trajectories are given as scatter plots (dots)
    the rest of the trajectories are given as continuous lines
    '''
    n = len(traj)

    colors = ['green', 'red', 'blue', 'orange']

    # for i in range(n - 2):
    #     shade = (i + 1) / (n - 2)
    #     colors.append(lighten_color('blue', shade))

    for i, x in enumerate(traj):
        if i < num_dots:
            plt.plot(x[0, :], x[1, :], 'o', color=colors[i], alpha=.5, label=labels[i])
        else:
            plt.plot(x[0, :], x[1, :], color=colors[i], alpha=.5, label=labels[i])

    plt.legend()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()


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
            x[:, t + 1] == A @ x[:, t] + B @ w[:, t],
            y[:, t] == C @ x[:, t] + v[:, t],
        ]

    prob = cp.Problem(obj, constr)
    prob.solve(verbose=True)

    x = np.array(x.value)
    w = np.array(w.value)
    v = np.array(v.value)

    return x, w, v


def cvxpy_manual(T, y, gamma, dt, mu, rho):
    x = cp.Variable(shape=(4, T + 1))
    w = cp.Variable(shape=(2, T))
    v = cp.Variable(shape=(2, T))
    s = cp.Variable(shape=(T))
    u = cp.Variable(shape=(T))
    z = cp.Variable(shape=(T))
    A, B, C = robust_kalman_setup(gamma, dt)

    obj = cp.sum_squares(w)
    obj += cp.sum([mu * (2 * u[t] + rho * z[t] ** 2) for t in range(T)])
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
            cp.norm(v[:, t]) <= s[t],
        ]

    cp.Problem(obj, constr).solve(verbose=True)

    x = np.array(x.value)
    w = np.array(w.value)
    v = np.array(v.value)
    return x, w, v


partial(jit, static_argnums=(2, 3, 4))


def simulate_fwd(w_mat, v_mat, T, gamma, dt, B_const):
    # def simulate(T, gamma, dt, sigma, p):
    A, B, C = robust_kalman_setup(gamma, dt, B_const)

    # generate random input and noise vectors
    # w = np.random.randn(2, T)
    # v = np.random.randn(2, T)

    x = jnp.zeros((4, T + 1))
    # x[:, 0] = [0, 0, 0, 0]
    y_mat = jnp.zeros((2, T))

    # add outliers to v
    # np.random.seed(0)
    # inds = np.random.rand(T) <= p
    # v[:, inds] = sigma * np.random.randn(2, T)[:, inds]

    # simulate the system forward in time
    for t in range(T):
        y_mat = y_mat.at[:, t].set(C.dot(x[:, t]) + v_mat[:, t])
        x = x.at[:, t + 1].set(A.dot(x[:, t]) + B.dot(w_mat[:, t]))

    # x_true = x.copy()
    # w_true = w_mat.copy()
    # return y, x_true, w_true, v_mat

    return y_mat, x


def get_x_w_true(theta, T, gamma, dt):
    A, B, C = robust_kalman_setup(gamma, dt)
    nu, no = 2, 2

    # extract (w, v)

    # theta = (w_0,...,w_{T-1},v_0,...,v_{T-1})

    # get y
    w = theta[: T * nu]
    v = theta[T * nu:]
    w_mat = jnp.reshape(w, (nu, T))
    v_mat = jnp.reshape(v, (no, T))
    # y_mat = simulate_fwd(w_mat, v_mat, T, gamma, dt)
    x = jnp.zeros((4, T + 1))
    y_mat = jnp.zeros((2, T))

    # simulate the system forward in time
    for t in range(T):
        y_mat = y_mat.at[:, t].set(C.dot(x[:, t]) + v_mat[:, t])
        x = x.at[:, t + 1].set(A.dot(x[:, t]) + B.dot(w_mat[:, t]))
    # y = jnp.ravel(y_mat.T)
    return x, w_mat


# @functools.partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8,))
# def single_q(theta, m, n, T, nx, nu, state_box, control_box, A_dynamics):
def single_q(theta, mu, rho, T, gamma, dt):
    """
    the observations, y_0,...,y_{T-1} are the parameters that change
    there are 6 blocks of constraints and the second one changes
    1. x_{t+1}=Ax_t+Bw_t
    2. y_t=Cx_t+v_t
    3. ...
    """
    nx, nu, no = 4, 2, 2
    single_len = nx + nu + no + 3
    nvars = single_len * T

    # extract (w, v)

    # theta = (w_0,...,w_{T-1},v_0,...,v_{T-1})

    # get y
    # w = theta[: T * nu]
    # v = theta[T * nu :]
    # w_mat = jnp.reshape(w, (nu, T))
    # v_mat = jnp.reshape(v, (no, T))
    # y_mat = simulate_fwd(w_mat, v_mat, T, gamma, dt)
    # y = jnp.ravel(y_mat.T)
    y = theta

    # c
    c = jnp.zeros(single_len * T)
    c = c.at[-2 * T: -T].set(2 * mu)

    # b
    b_dyn = jnp.zeros((T-1) * nx)
    # b_obs = jnp.zeros(T * no)
    b_obs = y

    # aux constraints
    n_aux = T
    b_aux = jnp.zeros(n_aux)

    # z_ineq constraints
    n_z_ineq = T
    b_z_ineq = rho * jnp.ones(n_z_ineq)

    # u_ineq constraints
    n_u_ineq = T
    b_u_ineq = jnp.zeros(n_u_ineq)

    # socp constraints
    n_socp = T * 3
    b_socp = jnp.zeros(n_socp)

    # get b
    b = jnp.hstack([b_dyn, b_obs, b_aux, b_z_ineq, b_u_ineq, b_socp])

    # q = jnp.zeros(n + m)
    # beq = jnp.zeros(T * nx)
    # beq = beq.at[:nx].set(A_dynamics @ theta)
    # b_upper = jnp.hstack([state_box * jnp.ones(T * nx), control_box * jnp.ones(T * nu)])
    # b_lower = jnp.hstack([state_box * jnp.ones(T * nx), control_box * jnp.ones(T * nu)])
    # b = jnp.hstack([beq, b_upper, b_lower])

    # q
    m = b.size
    q = jnp.zeros(m + nvars)
    q = q.at[:nvars].set(c)
    q = q.at[nvars:].set(b)
    # print('y', y)

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
    example = "robust_kalman"
    # folder = f"{orig_cwd}/outputs/{example}/aggregate_outputs/{datetime}"
    # data_yaml_filename = f"{folder}/data_setup_copied.yaml"
    data_yaml_filename = 'data_setup_copied.yaml'

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    static_dict = static_canon(
        setup_cfg['T'],
        setup_cfg['gamma'],
        setup_cfg['dt'],
        setup_cfg['mu'],
        setup_cfg['rho'],
        setup_cfg['B_const']
    )

    # get_q_single = partial(single_q,
    #                        mu=setup_cfg['mu'],
    #                        rho=setup_cfg['rho'],
    #                        T=setup_cfg['T'],
    #                        gamma=setup_cfg['gamma'],
    #                        dt=setup_cfg['dt'])

    # get_q = vmap(get_q_single, in_axes=0, out_axes=0)
    get_q = None

    """
    static_flag = True
    means that the matrices don't change across problems
    we only need to factor once
    """
    static_flag = True

    custom_visualize_fn_partial = partial(custom_visualize_fn, T=setup_cfg['T'])

    workspace = Workspace(run_cfg, static_flag, static_dict, example, get_q,
                          custom_visualize_fn=custom_visualize_fn_partial)

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
    - canonicalize according to whether we have states or not
    - extract information dependent on the setup
    """
    log.info("creating static canonicalization...")
    t0 = time.time()
    out_dict = static_canon(
        cfg.T,
        cfg.gamma,
        cfg.dt,
        cfg.mu,
        cfg.rho,
        cfg.B_const
    )

    t1 = time.time()
    log.info(f"finished static canonicalization - took {t1-t0} seconds")

    cones_dict = out_dict["cones_dict"]
    A_sparse, P_sparse = out_dict["A_sparse"], out_dict["P_sparse"]

    b, c = out_dict["b"], out_dict["c"]

    m, n = A_sparse.shape

    """
    save output to output_filename
    """
    # save to outputs/mm-dd-ss/... file
    # if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
    #     slurm_idx = os.environ["SLURM_ARRAY_TASK_ID"]
    #     output_filename = f"{os.getcwd()}/data_setup_slurm_{slurm_idx}"
    # else:
    #     output_filename = f"{os.getcwd()}/data_setup_slurm"
    output_filename = f"{os.getcwd()}/data_setup"
    """
    create scs solver object
    we can cache the factorization if we do it like this
    """

    data = dict(P=P_sparse, A=A_sparse, b=b, c=c)
    tol_abs = cfg.solve_acc_abs
    tol_rel = cfg.solve_acc_rel
    solver = scs.SCS(data,
                     cones_dict,
                     normalize=False,
                     scale=1,
                     adaptive_scale=False,
                     rho_x=1,
                     alpha=1,
                     acceleration_lookback=0,
                     eps_abs=tol_abs,
                     eps_rel=tol_rel)
    solve_times = np.zeros(N)
    x_stars = jnp.zeros((N, n))
    y_stars = jnp.zeros((N, m))
    s_stars = jnp.zeros((N, m))
    q_mat = jnp.zeros((N, m + n))

    """
    sample theta and get y for each problem
    """
    out = sample_theta(N, cfg.T, cfg.sigma, cfg.p, cfg.gamma, cfg.dt,
                       cfg.w_noise_var, cfg.y_noise_var, cfg.B_const)
    thetas_np, y_mat, x_trues, w_trues, y_mat_rotated, x_trues_rotated, w_trues_rot, angles = out
    thetas = jnp.array(thetas_np)

    batch_q = vmap(single_q, in_axes=(0, None, None, None, None, None), out_axes=(0))

    q_mat = batch_q(thetas, cfg.mu, cfg.rho, cfg.T, cfg.gamma, cfg.dt)

    scs_instances = []

    for i in range(N):
        log.info(f"solving problem number {i}")
        print(f"solving problem number {i}")

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

        if i % 1000:
            log.info(f"saving data... iteration {i}")
            jnp.savez(
                output_filename,
                thetas=thetas,
                x_stars=x_stars,
                y_stars=y_stars,
                s_stars=s_stars,
                q_mat=q_mat
            )

    # resave the data??
    # print('saving final data...', flush=True)
    log.info("saving final data...")
    jnp.savez(
        output_filename,
        thetas=thetas,
        x_stars=x_stars,
        y_stars=y_stars,
        s_stars=s_stars,
        q_mat=q_mat
    )

    # save solve times
    df_solve_times = pd.DataFrame(solve_times, columns=['solve_times'])
    df_solve_times.to_csv('solve_times.csv')

    # save angles
    df_angles = pd.DataFrame(angles, columns=['angles'])
    df_angles.to_csv('angles.csv')

    # print(f"finished saving final data... took {save_time-t0}'", flush=True)
    save_time = time.time()
    log.info(f"finished saving final data... took {save_time-t0}'")

    # save plot of first 5 solutions
    for i in range(5):
        plt.plot(x_stars[i, :])
    plt.savefig("x_stars.pdf")
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

    time_limit = cfg.dt * cfg.T
    ts, delt = np.linspace(0, time_limit, cfg.T-1, endpoint=True, retstep=True)

    os.mkdir("states_plots")
    os.mkdir("positions_plots")
    for i in range(5):
        x_state = x_stars[i, :cfg.T * 4]
        x1_kalman = x_state[0::4]
        x2_kalman = x_state[1::4]
        x_kalman = jnp.stack([x1_kalman, x2_kalman])

        # plot original
        # rotate back the output, x_kalman
        clockwise = False

        x_kalman_rotated_transpose = rotation_single(x_kalman.T, angles[i], clockwise)
        x_kalman_rotated = x_kalman_rotated_transpose.T

        # plot original
        plot_positions_overlay([x_trues[i, :, :-1], x_kalman_rotated, y_mat[i, :, :]],
                               ['True', 'KF recovery', 'Noisy'],
                               filename=f"positions_plots/positions_{i}.pdf")

        plot_positions_overlay([x_trues_rotated[i, :, :-1], x_kalman, y_mat_rotated[i, :, :]],
                               ['True', 'KF recovery', 'Noisy'],
                               filename=f"positions_plots/positions_{i}_rotated.pdf")


def custom_visualize_fn(x_primals, x_stars, x_no_learn, thetas, iterates, visual_path, T):
    """
    assume len(iterates) == 1 for now
        point is to compare no-learning vs learned for 20 iterations
    """
    assert len(iterates) == 1
    num = 5
    y_mat_rotated = jnp.reshape(thetas[:num, :], (num, T, 2))
    for i in range(5):
        titles = ['x_star', 'noisy']
        x_true_kalman = get_x_kalman_from_x_primal(x_stars[i, :], T)
        traj = [x_true_kalman, y_mat_rotated[i, :].T]

        for j in range(len(iterates)):
            iter = iterates[j]
            x_no_learn_kalman = get_x_kalman_from_x_primal(x_no_learn[i, iter, :], T)
            x_hat_kalman = get_x_kalman_from_x_primal(x_primals[i, iter, :], T)
            traj.append(x_no_learn_kalman)
            traj.append(x_hat_kalman)
            # titles.append(f"iterate {iter}")
            titles.append(f"no learning: {iter} iters")
            titles.append(f"learned: {iter} iters")

        plot_positions_overlay(traj, titles, filename=f"{visual_path}/positions_{i}_rotated.pdf")


def get_x_kalman_from_x_primal(x_primal, T):
    x_state = x_primal[:T * 4]
    # x_control = x_primal[T * 4: T * 6 - 2]
    x1_kalman = x_state[0::4]
    x2_kalman = x_state[1::4]
    x_kalman = jnp.stack([x1_kalman, x2_kalman])
    return x_kalman


def get_full_x(x0, x_w, y, T, Ad, Bd, rho):
    '''
    returns full x variable without redundant constraints
    '''
    nx, nu, no = 4, 2, 2
    x = jnp.zeros(nx)
    x_x = jnp.zeros(T*nx)
    x_v = jnp.zeros(T*no)
    x_s = jnp.zeros(T)
    x_z = jnp.zeros(T)
    x_u = jnp.zeros(T)
    curr_x = x0
    for i in range(T):
        curr_w = x_w[nu*i:nu*(i+1)]
        curr_x = Ad @ curr_x + Bd @ curr_w
        curr_y = y[no*i:no*(i+1)]
        x_pos = jnp.array([curr_x[0], curr_x[1]])
        curr_v = curr_y - x_pos

        curr_s = jnp.linalg.norm(curr_v)
        curr_z = jnp.min(jnp.array([curr_s, rho]))
        curr_u = curr_s - curr_z

        x_x = x_x.at[nx*i:nx*(i+1)].set(curr_x)
        x_v = x_v.at[no*i:no*(i+1)].set(curr_v)
        x_s = x_s.at[i].set(curr_s)
        x_u = x_u.at[i].set(curr_u)
        x_z = x_z.at[i].set(curr_z)

    x = jnp.concatenate([x_x, x_w, x_s, x_v, x_u, x_z])
    return x


def static_canon(T, gamma, dt, mu, rho, B_const):
    """
    variables
    (x_t, w_t, s_t, v_t,  u_t, z_t) in (nx + nu + no + 3)
    (nx,  nu,  1,   no,   no, 1, 1)
    min sum_{i=0}^{T-1} ||w_t||_2^2 + mu (u_t+rho*z_t^2)
        s.t. x_{t+1} = Ax_t + Bw_t  t=0,...,T-2 (dyn)
             y_t = Cx_t + v_t       t=0,...,T-1 (obs)
             u_t + z_t = s_t        t=0,...,T-1 (aux)
             z_t <= rho             t=0,...,T-1 (z ineq)
             u_t >= 0               t=0,...,T-1 (u ineq)
             ||v_t||_2 <= s_t       t=0,...,T-1 (socp)
    (x_0, ..., x_{T-1})
    (y_0, ..., y_{T-1})
    (w_0, ..., w_{T-2})
    (v_0, ..., v_{T-1})
    """
    # nx, nu, no don't change
    nx, nu, no = 4, 2, 2

    # to make indexing easier
    single_len = nx + nu + no + 3
    nvars = single_len * T
    w_start = nx * T
    s_start = w_start + nu * T
    v_start = s_start + T
    u_start = v_start + no * T
    z_start = u_start + T

    assert z_start + T == single_len * T

    # get A, B, C
    Ad, Bd, C = robust_kalman_setup(gamma, dt, B_const)

    # Quadratic objective
    P = np.zeros((single_len, single_len))
    P[nx: nx + nu, nx: nx + nu] = np.eye(nu)
    P[-1, -1] = mu * rho
    # P_sparse = sparse.kron(sparse.eye(T), P)
    P_sparse = 2*sparse.kron(P, sparse.eye(T))

    # Linear objective
    c = np.zeros(single_len * T)
    c[-2 * T: -T] = 2 * mu

    # dyn constraints
    Ax = sparse.kron(sparse.eye(T), -sparse.eye(nx)) + sparse.kron(
        sparse.eye(T, k=-1), Ad
    )
    Ax = Ax[nx:, :]

    # Bw = sparse.kron(sparse.eye(T), Bd)
    Bw = sparse.kron(sparse.eye(T), Bd)
    bw = Bw.todense()
    bw = bw[:(T-1)*nx, :]
    A_dyn = np.zeros(((T-1) * nx, nvars))
    A_dyn[:, :w_start] = Ax.todense()

    A_dyn[:, w_start:s_start] = bw  # Bw.todense()
    b_dyn = np.zeros((T-1) * nx)

    # obs constraints
    Cx = np.kron(np.eye(T), C)

    Iv = np.kron(np.eye(2), np.eye(T))

    A_obs = np.zeros((T * no, nvars))
    A_obs[:, :w_start] = Cx

    A_obs[:, v_start:u_start] = Iv
    # b_obs will be updated by the parameter stack(y_1, ..., y_T)
    b_obs = np.zeros(T * no)

    # aux constraints
    n_aux = T
    A_aux = np.zeros((n_aux, nvars))
    A_aux[:, s_start:v_start] = -np.eye(T)
    A_aux[:, u_start:z_start] = np.eye(T)
    A_aux[:, z_start:] = np.eye(T)
    b_aux = np.zeros(n_aux)

    # z_ineq constraints
    n_z_ineq = T
    A_z_ineq = np.zeros((n_z_ineq, nvars))
    A_z_ineq[:, z_start:] = np.eye(n_z_ineq)
    b_z_ineq = rho * np.ones(n_z_ineq)
    # pdb.set_trace()

    # u_ineq constraints
    n_u_ineq = T
    A_u_ineq = np.zeros((n_u_ineq, nvars))
    A_u_ineq[:, u_start:z_start] = -np.eye(n_u_ineq)
    b_u_ineq = np.zeros(n_u_ineq)

    # socp constraints
    n_socp = T * 3
    A_socp = np.zeros((n_socp, nvars))
    for i in range(T):
        A_socp[3 * i, s_start + i] = -1
        A_socp[
            3 * i + 1: 3 * i + 3, v_start + 2 * i: v_start + 2 * (i + 1)
        ] = -np.eye(2)

    b_socp = np.zeros(n_socp)

    # stack A
    A_sparse = sparse.vstack([A_dyn, A_obs, A_aux, A_z_ineq, A_u_ineq, A_socp])

    # get b
    b = np.hstack([b_dyn, b_obs, b_aux, b_z_ineq, b_u_ineq, b_socp])

    q_array = [3 for i in range(T)]  # np.ones(T) * 3
    # q_array_jax = jnp.array(q_array)
    cones = dict(z=T * (1 + nx + no) - nx, l=n_z_ineq + n_u_ineq, q=q_array)
    cones_array = jnp.array([cones["z"], cones["l"]])
    cones_array = jnp.concatenate([cones_array, jnp.array(cones["q"])])

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
        cones_dict=cones,
        cones_array=cones_array,
        A_sparse=A_sparse,
        P_sparse=P_sparse,
        b=b,
        c=c,
        A_dynamics=Ad,
        Bd=Bd
    )
    return out_dict


def robust_kalman_setup(gamma, dt, B_const):
    A = jnp.zeros((4, 4))
    B = jnp.zeros((4, 2))
    C = jnp.zeros((2, 4))

    A = A.at[0, 0].set(1)
    A = A.at[1, 1].set(1)
    A = A.at[0, 2].set((1 - gamma * dt / 2) * dt)
    A = A.at[1, 3].set((1 - gamma * dt / 2) * dt)
    A = A.at[2, 2].set(1 - gamma * dt)
    A = A.at[3, 3].set(1 - gamma * dt)

    B = B.at[0, 0].set(dt**2 / 2)
    B = B.at[1, 1].set(dt**2 / 2)
    B = B.at[2, 0].set(dt)
    B = B.at[3, 1].set(dt)

    C = C.at[0, 0].set(1)
    C = C.at[1, 1].set(1)

    return A, B * B_const, C


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
