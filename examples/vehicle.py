import functools
import hydra
import cvxpy as cp
from jax import linear_transpose
import pandas as pd
from pandas import read_csv
from l2ws.scs_problem import SCSinstance, scs_jax, ruiz_equilibrate
import numpy as np
import pdb
from l2ws.launcher import Workspace
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
from utils.mpc_utils import static_canon
log = logging.getLogger(__name__)


def setup_probs(cfg):
    N = cfg.N_train + cfg.N_test
    control_box = np.array([cfg.Fy_box, cfg.Mx_box, cfg.Mz_box])

    '''
    sample N theta vectors
    theta = (v0, x0, {y_t}, {delta_t})
    OR
    theta = (v0, x0, {y_t}, delta_0, rate_delta)
    '''
    thetas, specifics_dict = sample_thetas(cfg)

    '''
    get loose canonicalization
    P will be correct
    A, c, b correct shape
    ** A: correct except for Ad, Bd part
    ** b: x0 and E delta parts need to be updated
    ** c: reference trajectory part updated (linear term for controls zero)
    '''
    T, nx, nu, dt = cfg.T, 4, 3, cfg.dt
    state_box = np.inf
    control_box = np.array([cfg.Fy_box * cfg.Fy_factor,
                            cfg.Mx_box * cfg.Mx_factor,
                            cfg.Mz_box * cfg.Mz_factor])
    Q_vec = np.array([cfg.slip_penalty, cfg.yaw_penalty,
                     cfg.roll_penalty, cfg.roll_rate_penalty])
    QT_vec = np.array([cfg.slip_penalty, cfg.yaw_penalty,
                      cfg.roll_penalty, cfg.roll_rate_penalty])
    R_vec = np.array([cfg.Fy_penalty / (cfg.Fy_factor ** 2),
                      cfg.Mx_penalty / (cfg.Mx_factor ** 2),
                      cfg.Mz_penalty / (cfg.Mz_factor ** 2)])
    # delta_control_box = cfg.control_rate_lim
    delta_control_box = np.array([
        cfg.control_rate_lim * cfg.Fy_factor,
        cfg.control_rate_lim * cfg.Mx_factor,
        cfg.control_rate_lim * cfg.Mz_factor
    ])

    static_dict = static_canon(T, nx, nu,
                               state_box,
                               control_box,
                               Q_vec,
                               QT_vec,
                               R_vec,
                               Ad=None,
                               Bd=None,
                               delta_control_box=delta_control_box)
    M_tilde = jnp.array(static_dict['M'])
    c_tilde, b_tilde = static_dict['c'], static_dict['b']
    q_tilde = jnp.concatenate([c_tilde, b_tilde])
    A_sparse = static_dict['A_sparse']
    m, n = A_sparse.shape
    '''
    canonicalize all at once with vmap
    '''
    get_M_q_single_partial = functools.partial(
        get_single_M_q, M_tilde=M_tilde, q_tilde=q_tilde, T=T, m=m, n=n, dt=dt,
        u_box=jnp.array(control_box),
        Fy_factor=cfg.Fy_factor, Mx_factor=cfg.Mx_factor,
        Mz_factor=cfg.Mz_factor)
    get_M_q_batch = vmap(get_M_q_single_partial,
                         in_axes=(0,), out_axes=(0, 0,))
    M_tensor, q_mat = get_M_q_batch(thetas)

    solve_times = np.zeros(N)
    x_stars = jnp.zeros((N, n))
    y_stars = jnp.zeros((N, m))
    s_stars = jnp.zeros((N, m))

    tol = cfg.solve_acc

    scs_instances = []
    cones_array_np = np.array(static_dict['cones_array'])
    cones_dict = dict(z=int(cones_array_np[0]), l=int(cones_array_np[1]))

    matrix_invs = jnp.zeros(((N, m+n, m+n)))
    factor1s = jnp.zeros(((N, m+n, m+n)))
    factor2s = jnp.zeros(((N, m+n, m+n)))

    for i in range(N):
        log.info(f"solving problem number {i}")

        P_sparse = csc_matrix(np.array(M_tensor[i, :n, :n]))
        A_sparse = -csc_matrix(np.array(M_tensor[i, n:, :n]))
        b_np = np.array(q_mat[i, n:])
        c_np = np.array(q_mat[i, :n])
        '''
        retrieve (P, A, c, b) to pass to solver
        '''
        data = {'P': P_sparse, 'A': A_sparse,
                'b': b_np, 'c': c_np}
        solver = scs.SCS(data, cones_dict, eps_abs=tol, eps_rel=tol)
        scs_instance = SCSinstance(
            data, solver, manual_canon=True)

        scs_instances.append(scs_instance)
        x_stars = x_stars.at[i, :].set(scs_instance.x_star)
        y_stars = y_stars.at[i, :].set(scs_instance.y_star)
        s_stars = s_stars.at[i, :].set(scs_instance.s_star)
        solve_times[i] = scs_instance.solve_time

        '''
        factor (M + I) or take inverse for l2ws use
        '''
        M_I = M_tensor[i, :, :] + jnp.eye(m + n)

        factor1, factor2 = jsp.linalg.lu_factor(M_I)
        factor1s = factor1s.at[i, :, :].set(factor1)
        factor2s = factor2s.at[i, :, :].set(factor2)

        '''
        check against cvxpy
        T, nx, nu, dt = input_dict['T'], 4, 3, input_dict['dt']
        A, B, E = input_dict['A'], input_dict['B'], input_dict['E']
        delta0, delta_rate = input_dict['delta0'], input_dict['delta_rate']
        x0, u_prev = input_dict['x0'], input_dict['u_prev']
        box_control = input_dict['box_control']
        box_delta_control = input_dict['box_delta_control']
        ref_traj = input_dict['rej_traj']
        Q, R = input_dict['Q'], input_dict['R']
        '''
        # uncomment for cvxpy canon check
        # delta0 = specifics_dict['delta0_mat'][0, :]
        # delta_rate = specifics_dict['delta_rate_mat'][0, :]
        # x0 = specifics_dict['x0_mat'][0, :]
        # u_prev = specifics_dict['u_prev_mat'][0, :]
        # ref_traj_full = specifics_dict['ref_traj_mat'][0, :]
        # ref_traj_mat = np.reshape(ref_traj_full, (T, 3))

        # vel = specifics_dict['vels'][0, :]
        # Ad, Bd, Ed = get_ABE(vel, dt, cfg.Fy_factor,
        #                      cfg.Mx_factor, cfg.Mz_factor)
        # cvxpy_dict = {'T': T, 'A': Ad, 'B': Bd, 'E': Ed, 'dt': dt,
        #               'delta0': delta0,
        #               'delta_rate': delta_rate,
        #               'x0': x0,
        #               'u_prev': u_prev,
        #               'box_control': control_box,
        #               'box_delta_control': cfg.control_rate_lim,
        #               'ref_traj': ref_traj_mat,
        #               'Q': np.diag(Q_vec),
        #               'R': np.diag(R_vec)}
        # x_sol, u_sol = cvxpy_check(cvxpy_dict)

        '''
        try with our implementation
        '''
        # P_jax = jnp.array(P_sparse.todense())
        # A_jax = jnp.array(A_sparse.todense())
        # c_jax, b_jax = jnp.array(c_np), jnp.array(b_np)
        # data = dict(P=P_jax, A=A_jax, b=b_jax, c=c_jax, cones=cones_dict)
        # data['x'] = x_stars[i, :]
        # data['y'] = y_stars[i, :]
        # x_jax, y_jax, s_jax = scs_jax(data, iters=1000)
        # pdb.set_trace()

        # plt.plot(x_stars[0, :T*nx], label='solution')
        # plt.plot(thetas[0, 10:], label='reference')

        # plt.show()
        # pdb.set_trace()

    '''
    save (thetas, x_stars, y_stars, factorizations)
    '''

    if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
        slurm_idx = os.environ["SLURM_ARRAY_TASK_ID"]
        output_filename = f"{os.getcwd()}/data_setup_slurm_{slurm_idx}"
    else:
        output_filename = f"{os.getcwd()}/data_setup_slurm"

    log.info('saving final data...')
    t0 = time.time()
    if cfg.linear_system_solve == 'inverse':
        jnp.savez(output_filename,
                  thetas=thetas,
                  x_stars=x_stars,
                  y_stars=y_stars,
                #   matrix_invs=matrix_invs
                  )
    elif cfg.linear_system_solve == 'factor':
        jnp.savez(output_filename,
                  thetas=thetas,
                  x_stars=x_stars,
                  y_stars=y_stars,
                #   factors1s=factor1s,
                #   factors2s=factor2s
                  )
    # print(f"finished saving final data... took {save_time-t0}'", flush=True)
    save_time = time.time()
    log.info(f"finished saving final data... took {save_time-t0}'")
    # plt.plot(x_stars[0, :])
    # plt.plot(x_stars[1, :])
    # plt.plot(x_stars[2, :])
    # plt.show()


def cvxpy_check(input_dict):
    '''
    use to verify canonicalization and check the optiimial solns match
    '''
    T, nx, nu, dt = input_dict['T'], 4, 3, input_dict['dt']
    A, B, E = input_dict['A'], input_dict['B'], input_dict['E']
    delta0, delta_rate = input_dict['delta0'], input_dict['delta_rate']
    x0, u_prev = input_dict['x0'], input_dict['u_prev']
    box_control = input_dict['box_control']
    box_delta_control = input_dict['box_delta_control']
    ref_traj = input_dict['ref_traj']
    Q, R = input_dict['Q'], input_dict['R']

    x = cp.Variable((T, nx))
    u = cp.Variable((T, nu))
    obj = 0
    # pdb.set_trace()
    for i in range(T):
        full_ref = np.array(
            [ref_traj[i, 0], ref_traj[i, 1], 0, ref_traj[i, 2]])
        diff = x[i, :] - full_ref
        obj += cp.quad_form(diff, Q)
        obj += cp.quad_form(u[i, :], R)
    constraints = []

    # first dynamics constraint
    constraints.append(x[0, :] == A @ x0 + B @ u[0, :] + E * delta0)

    # first delta box control constraint
    constraints.append(cp.abs(u[0, :] - u_prev) <= box_delta_control)
    # constraints.append(cp.abs(u[0, 1] - ) <= box_control[1])
    # constraints.append(cp.abs(u[0, 2]) <= box_control[2])

    for i in range(1, T):
        # dynamics
        drift = delta0 + i * dt * delta_rate
        constraints.append(x[i, :] == A @ x[i-1, :] + B @ u[i, :] + E * drift)

        # box delta control constraints
        # constraints.append(cp.abs(u[i, 0]) <= box_control[0])
        # constraints.append(cp.abs(u[i, 1]) <= box_control[1])
        # constraints.append(cp.abs(u[i, 2]) <= box_control[2])
        constraints.append(cp.abs(u[i, :]) <= box_control)

    for i in range(T):
        # box control constraints
        # constraints.append(cp.abs(u[i, 0]) <= box_control[0])
        # constraints.append(cp.abs(u[i, 1]) <= box_control[1])
        # constraints.append(cp.abs(u[i, 2]) <= box_control[2])
        constraints.append(cp.abs(u[i, :]) <= box_control)

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS, verbose=True)
    x_opt = x.value
    u_opt = u.value
    return x_opt, u_opt


def run(run_cfg):
    '''
    retrieve data for this config
    theta is all of the following
    theta = (ret, pen_risk, pen_hold, pen_trade, w0)

    Sigma is constant

     just need (theta, factor, u_star), Pi
    '''
    # todo: retrieve data and put into a nice form - OR - just save to nice form

    '''
    create workspace
    needs to know the following somehow -- from the run_cfg
    1. nn cfg
    2. (theta, factor, u_star)_i=1^N
    3. Pi

    2. and 3. are stored in data files and the run_cfg holds the location

    it will create the l2a_model
    '''
    datetime = run_cfg.data.datetime
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'vehicle'
    data_yaml_filename = f"{orig_cwd}/outputs/{example}/aggregate_outputs/{datetime}/data_setup_copied.yaml"

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    nx, nu = 4, 3
    T, nx, nu, dt = setup_cfg['T'], 4, 3, setup_cfg['dt']
    state_box = np.inf
    control_box = np.array([setup_cfg['Fy_box'] * setup_cfg['Fy_factor'],
                            setup_cfg['Mx_box'] * setup_cfg['Mx_factor'],
                            setup_cfg['Mz_box'] * setup_cfg['Mz_factor']])
    Q_vec = np.array([setup_cfg['slip_penalty'], setup_cfg['yaw_penalty'],
                     setup_cfg['roll_penalty'], setup_cfg['roll_rate_penalty']])
    QT_vec = np.array([setup_cfg['slip_penalty'], setup_cfg['yaw_penalty'],
                      setup_cfg['roll_penalty'], setup_cfg['roll_rate_penalty']])
    R_vec = np.array([setup_cfg['Fy_penalty'] / (setup_cfg['Fy_factor'] ** 2),
                      setup_cfg['Mx_penalty'] / (setup_cfg['Mx_factor'] ** 2),
                      setup_cfg['Mz_penalty'] / (setup_cfg['Mz_factor'] ** 2)])
    delta_control_box = np.array([
        setup_cfg['control_rate_lim'] * setup_cfg['Fy_factor'],
        setup_cfg['control_rate_lim'] * setup_cfg['Mx_factor'],
        setup_cfg['control_rate_lim'] * setup_cfg['Mz_factor']
    ])

    static_dict = static_canon(T, nx, nu,
                               state_box,
                               control_box,
                               Q_vec,
                               QT_vec,
                               R_vec,
                               Ad=None,
                               Bd=None,
                               delta_control_box=delta_control_box)
    M_tilde = jnp.array(static_dict['M'])
    c_tilde, b_tilde = static_dict['c'], static_dict['b']
    q_tilde = jnp.concatenate([c_tilde, b_tilde])
    A_sparse = static_dict['A_sparse']
    m, n = A_sparse.shape
    '''
    canonicalize all at once with vmap
    '''
    get_M_q_single_partial = functools.partial(
        get_single_M_q, M_tilde=M_tilde, q_tilde=q_tilde,
        T=T, m=m, n=n, dt=dt,
        u_box=jnp.array(control_box),
        Fy_factor=setup_cfg['Fy_factor'], Mx_factor=setup_cfg['Mx_factor'],
        Mz_factor=setup_cfg['Mz_factor'])
    get_M_q_batch = vmap(get_M_q_single_partial,
                         in_axes=(0,), out_axes=(0, 0,))

    # A_dynamics = static_dict['A_dynamics']
    static_flag = False
    workspace = Workspace(run_cfg, static_flag, static_dict, 'vehicle', get_M_q_batch)

    '''
    run the workspace
    '''
    workspace.run()


def sample_thetas(cfg):
    N, T = cfg.N_train + cfg.N_test, cfg.T

    # convert degrees to radians
    const = np.pi / 180
    slip_box, yaw_box = const * cfg.slip_box_deg, const * cfg.yaw_box_deg
    roll_box = const * cfg.roll_box_deg
    roll_rate_box = const * cfg.roll_rate_box_deg
    u_box = np.array([cfg.Fy_box, cfg.Mx_box, cfg.Mz_box])
    delta_box = const * cfg.delta0_box_deg
    delta_rate_box = const * cfg.delta_rate_box_deg

    # sample velocity
    vels = cfg.v_min + (cfg.v_max - cfg.v_min) * np.random.rand(N, 1)

    # sample initial state
    slips = (2 * np.random.rand(N, T+1) - 1) * slip_box
    yaws = (2 * np.random.rand(N, T+1) - 1) * yaw_box
    rolls = (2 * np.random.rand(N, T+1) - 1) * roll_box
    roll_rates = (2 * np.random.rand(N, T+1) - 1) * roll_rate_box
    x0_mat = np.hstack([slips[:, 0:1], yaws[:, 0:1],
                       rolls[:, 0:1], roll_rates[:, 0:1]])

    # sample reference trajectory
    # ref_traj_mat = np.hstack(
    #     [slips[:, 1:], yaws[:, 1:], rolls[:, 1:], roll_rates[:, 1:]])
    ref_traj_mat = np.zeros((N, 3*T))
    for i in range(T):
        ref_traj_mat[:, 3*i:3*(i+1)] = np.hstack([slips[:, i+1:i+2], yaws[:, i+1:i+2],
                                                  roll_rates[:, i+1:i+2]])

    # sample u_prev_mat
    Fys = (2 * np.random.rand(N, 1) - 1) * cfg.Fy_box * cfg.Fy_factor
    Mxs = (2 * np.random.rand(N, 1) - 1) * cfg.Mx_box * cfg.Mx_factor
    Mzs = (2 * np.random.rand(N, 1) - 1) * cfg.Mz_box * cfg.Mz_factor
    u_prev_mat = np.hstack([Fys, Mxs, Mzs])

    # sample steering
    delta0_mat = (2 * np.random.rand(N, 1) - 1) * delta_box
    delta_rate_mat = (2 * np.random.rand(N, 1) - 1) * delta_rate_box

    thetas = jnp.hstack(
        [vels, x0_mat, u_prev_mat, delta0_mat, delta_rate_mat, ref_traj_mat])
    specifics = {'vels': vels,
                 'x0_mat': x0_mat,
                 'u_prev_mat': u_prev_mat,
                 'delta0_mat': delta0_mat,
                 'delta_rate_mat': delta_rate_mat,
                 'ref_traj_mat': ref_traj_mat
                 }
    return thetas, specifics


# def get_single_M_q(theta, dt, slip_pen, yaw_pen,
#                    roll_pen, roll_rate_pen,
#                    Fy_box, Mx_box, Mz_box,
#                    with_states=True):
def get_single_M_q(theta, M_tilde, q_tilde, T, m, n, dt, u_box, Fy_factor, Mx_factor, Mz_factor):
    '''
    first extract what theta means
    '''
    nx, nu, state_box = 4, 3, np.inf

    vel, x0 = theta[0], theta[1:1+nx]
    u_prev = theta[1+nx:1+nx+nu]
    delta0 = theta[1+nx+nu:2+nx+nu]
    delta_rate = theta[2+nx+nu:3+nx+nu]
    ref_traj = theta[3+nx+nu:]

    '''
    second get A(v), B(v), E(v)
    '''
    Ad, Bd, Ed = get_ABE(vel, dt, Fy_factor, Mx_factor, Mz_factor)

    '''
    second put the parameters into generic MPC form
    '''
    # for mQ_vec, QT_vec, R_vec
    # Q_vec = jnp.array([slip_pen, yaw_pen, roll_pen, roll_rate_pen])
    # QT_vec = jnp.array([slip_pen, yaw_pen, roll_pen, roll_rate_pen])
    # R_vec = jnp.array([Fy_box, Mx_box, Mz_box])
    # static_dict = static_canon(T, nx, nu,
    #                            state_box,
    #                            control_box,
    #                            Q_vec,
    #                            QT_vec,
    #                            R_vec,
    #                            Ad=None,
    #                            Bd=None,
    #                            delta_control_box=None)

    P, A = M_tilde[:n, :n], -M_tilde[n:, :n]
    c, b = q_tilde[:n], q_tilde[n:]

    '''
    update b
    '''
    deltas = delta0 + delta_rate * dt * jnp.arange(T)
    # b = b.at[:T*nx].set(jnp.kron(Ed, deltas))
    b = b.at[:T*nx].set(jnp.kron(deltas, Ed))

    # x0 is a part of b also
    b = b.at[:nx].set(b[:nx] + Ad @ x0)

    '''
    u_prev comes in as well 
    (T * nx) linear dynamics constraints
    (2 * T * nu) box control constraints
    + T * nu lower control diff box constraints
    '''
    start1 = T * nx + 2 * T * nu
    start2 = T * nx + 3 * T * nu
    b = b.at[start1:start1 + nu].set(u_prev + u_box)
    b = b.at[start2:start2 + nu].set(-u_prev + u_box)

    '''
    update c
    -- no linear part for the controls
    -- linear part of states is P_state @ ref_stacked
    ref_stacked = vstack(x_1, ..., x_T)
    '''

    #P_state = P[:T*nx, :T*nx]
    # c = c.at[:T*nx].set(-2 * P_state @ ref_traj)
    # take only states 1, 2, 4
    # Quadratic objective
    # P_sparse = sparse.block_diag(
    #     [sparse.kron(sparse.eye(T-1), Q), QT, sparse.kron(sparse.eye(T), R)],
    #     format="csc",
    # )
    P_state = P[:nx, :nx]
    PC = jnp.diag(jnp.array([P_state[0, 0], P_state[1, 1], 0, P_state[3, 3]]))
    PCref = jnp.kron(jnp.eye(T), PC)

    ref_traj_reshape = jnp.reshape(ref_traj, (T, 3))
    ref_traj_full = jnp.zeros((T, 4))
    ref_traj_full = ref_traj_full.at[:, :2].set(ref_traj_reshape[:, :2])
    ref_traj_full = ref_traj_full.at[:, 3].set(ref_traj_reshape[:, 3])
    ref_traj_full_vec = jnp.ravel(ref_traj_full)
    #
    c = c.at[:T*nx].set(-1 * PCref @ ref_traj_full_vec)

    '''
    update A
    '''
    Ax = jnp.kron(jnp.eye(T + 1), -jnp.eye(nx)) + jnp.kron(
        jnp.eye(T + 1, k=-1), Ad
    )
    Ax = Ax[nx:, nx:]

    Bu = jnp.kron(
        jnp.eye(T), Bd
    )
    Anew = jnp.hstack([Ax, Bu])

    A = A.at[:T * nx, :].set(Anew)

    M = jnp.zeros((m + n, m + n))
    P_factor = P.max()
    M = M.at[:n, :n].set(P / P_factor)
    M = M.at[n:, :n].set(-A)
    M = M.at[:n, n:].set(A.T)
    q = jnp.concatenate([c / P_factor, b])
    return M, q


def sample_parameters(N, T):
    thetas = np.zeros(())
    return Ad_tensor, Bd_tensor, thetas


def get_ABE(v, dt, Fy_factor, Mx_factor, Mz_factor):
    Ac = jnp.array([
        [-265.3955/v, -1+110.0726/v**2, -72.2979/v, -8.9103/v],
        [32.1661, -174.2618/v, 0.2896, .0357],
        [0, 0, 0, 1],
        [-185.4271, 76.9058/v, -164.2498, -20.2428]
    ])
    Bc = jnp.array([
        [95.2569/v, 0.000627071/v/Fy_factor, 0.000438123 /
            v/Mx_factor, -0.00000251207/v/Mz_factor],
        [41.7399, -0.00000251207/Fy_factor, -
            0.00000175514/Mx_factor, 0.000181167/Mz_factor],
        [0, 0, 0, 0],
        [66.5543, 0.000438123/Fy_factor, 0.000995349 /
            Mx_factor, -0.00000175514/Mz_factor]
    ])
    Bd = Bc[:, 1:] * dt
    Ed = Bc[:, 0] * dt
    Ad = jnp.eye(4) + Ac * dt

    return Ad, Bd, Ed
