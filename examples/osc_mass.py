import functools
import hydra
from l2ws.scs_problem import SCSinstance
import numpy as np
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
from jax import vmap
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
})
log = logging.getLogger(__name__)


def single_q(theta, m, n, T, nx, nu, state_box, control_box, A_dynamics):
    q = jnp.zeros(n + m)
    beq = jnp.zeros(T * nx)
    beq = beq.at[:nx].set(A_dynamics @ theta)
    b_upper = jnp.hstack(
        [state_box*jnp.ones(T * nx), control_box*jnp.ones(T * nu)])
    b_lower = jnp.hstack(
        [state_box*jnp.ones(T * nx), control_box*jnp.ones(T * nu)])
    b = jnp.hstack([beq, b_upper, b_lower])
    q = q.at[n:].set(b)
    return q


# def get_q_mat_control_box_only(thetas, m, n, T, nx, nu,
#                                state_box, control_box,
#                                QB, vecc_gen):
#     N, nx = thetas.shape
#     q_mat = control_box*jnp.ones((N, n + m))
#     for i in range(N):
#         c = QB @ thetas[i, :]
#         b = control_box*jnp.ones(m)
#         if state_box != 'inf':
#             start = 2*T*nu
#             rhs = vecc_gen @ thetas[i, :]
#             b = b.at[start:start + T*nx].set(state_box - rhs)
#             b = b.at[start + T*nx:start + 2 *
#                      T*nx].set(state_box + rhs)
#         q_mat = q_mat.at[i, :n].set(c)
#     return q_mat


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
    example = 'osc_mass'
    folder = f"{orig_cwd}/outputs/{example}/aggregate_outputs/{datetime}"
    data_yaml_filename = f"{folder}/data_setup_copied.yaml"

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    T, control_box = setup_cfg['T'], setup_cfg['control_box']
    state_box = setup_cfg['state_box']
    nx, nu = setup_cfg['nx'], setup_cfg['nu']
    Q_val, QT_val = setup_cfg['Q_val'], setup_cfg['QT_val']
    R_val = setup_cfg['R_val']

    Ad, Bd = oscillating_masses_setup(nx, nu)

    static_dict = static_canon(T, nx, nu,
                               state_box,
                               control_box,
                               Q_val,
                               QT_val,
                               R_val,
                               Ad=Ad,
                               Bd=Bd)
    A_sparse = static_dict['A_sparse']
    m, n = A_sparse.shape

    get_q_single = functools.partial(single_q,
                                     m=m,
                                     n=n,
                                     T=T,
                                     nx=nx,
                                     nu=nu,
                                     state_box=state_box,
                                     control_box=control_box,
                                     A_dynamics=Ad)
    get_q = vmap(get_q_single, in_axes=0, out_axes=0)

    '''
    static_flag = True
    means that the matrices don't change across problems
    we only need to factor once
    '''
    static_flag = True
    workspace = Workspace(run_cfg, static_flag, static_dict, example, get_q)

    '''
    run the workspace
    '''
    workspace.run()


def setup_probs(setup_cfg):
    print('entered osc mass setup', flush=True)
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test

    '''
    create the dynamics depending on the system
    '''
    Ad, Bd = oscillating_masses_setup(cfg.nx, cfg.nu)

    '''
    - canonicalize according to whether we have states or not
    - extract information dependent on the setup
    '''
    log.info('creating static canonicalization...')
    t0 = time.time()
    out_dict = static_canon(cfg.T, cfg.nx, cfg.nu,
                            cfg.state_box,
                            cfg.control_box,
                            cfg.Q_val,
                            cfg.QT_val,
                            cfg.R_val,
                            Ad=Ad,
                            Bd=Bd)

    t1 = time.time()
    log.info(f"finished static canonicalization - took {t1-t0} seconds")

    cones_array = out_dict['cones_array']
    A_sparse, P_sparse = out_dict['A_sparse'], out_dict['P_sparse']

    '''
    if with_states, b is updated
    if w/out states, c is updated
    '''
    b, c = out_dict['b'], out_dict['c']

    m, n = A_sparse.shape
    cones_dict = dict(z=int(cones_array[0]), l=int(cones_array[1]))

    '''
    save output to output_filename
    '''
    # save to outputs/mm-dd-ss/... file
    if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
        slurm_idx = os.environ["SLURM_ARRAY_TASK_ID"]
        output_filename = f"{os.getcwd()}/data_setup_slurm_{slurm_idx}"
    else:
        output_filename = f"{os.getcwd()}/data_setup_slurm"
    '''
    create scs solver object
    we can cache the factorization if we do it like this
    '''

    data = dict(P=P_sparse, A=A_sparse, b=b, c=c)
    tol = cfg.solve_acc
    solver = scs.SCS(data, cones_dict, eps_abs=tol, eps_rel=tol)
    solve_times = np.zeros(N)
    x_stars = jnp.zeros((N, n))
    y_stars = jnp.zeros((N, m))
    s_stars = jnp.zeros((N, m))
    q_mat = jnp.zeros((N, m+n))

    '''
    initialize x_init over all the problems
    '''
    # x_init_mat initialized uniformly between x_init_box*[-1,1]
    x_init_mat = cfg.x_init_box * (2 * np.random.rand(N, cfg.nx) - 1)

    scs_instances = []
    for i in range(N):
        infeasible = True
        while infeasible:
            log.info(f"solving problem number {i}")
            b[:cfg.nx] = Ad @ x_init_mat[i, :]

            # manual canon
            manual_canon_dict = {'P': P_sparse, 'A': A_sparse,
                                 'b': b, 'c': c,
                                 'cones': cones_dict}
            scs_instance = SCSinstance(
                manual_canon_dict, solver, manual_canon=True)

            '''
            check feasibility
            if infeasible, resample
            '''
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

    # resave the data??
    # print('saving final data...', flush=True)
    log.info('saving final data...')
    t0 = time.time()
    jnp.savez(output_filename,
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
    plt.savefig('opt_solutions.pdf')

    # save plot of first 5 parameters
    for i in range(5):
        plt.plot(x_init_mat[i, :])
    plt.savefig('thetas.pdf')


def static_canon_osqp(T, nx, nu, state_box, control_box, Q_val, QT_val, R_val, Ad=None, Bd=None):
    if np.isscalar(Q_val):
        Q = Q_val * np.eye(nx)
    else:
        Q = Q_val
    if np.isscalar(Q_val):
        QT = QT_val * np.eye(nx)
    else:
        QT = QT_val
    if np.isscalar(R_val):
        R = R_val * np.eye(nu)
    else:
        R = R_val
    q = np.zeros(nx)  # np.random.normal(size=(nx))#
    qT = np.zeros(nx)

    if Ad is None and Bd is None:
        Ad = .1 * np.random.normal(size=(nx, nx))
        Bd = .1 * np.random.normal(size=(nx, nu))

    # Quadratic objective
    P = sparse.block_diag(
        [sparse.kron(sparse.eye(T-1), Q), QT, sparse.kron(sparse.eye(T), R)],
        format="csc",
    )

    # Linear objective
    c = np.hstack([np.kron(np.ones(T-1), q), qT, np.zeros(T * nu)])

    # Linear dynamics
    Ax = sparse.kron(sparse.eye(T + 1), -sparse.eye(nx)) + sparse.kron(
        sparse.eye(T + 1, k=-1), Ad
    )
    Ax = Ax[nx:, nx:]
    Bu = sparse.kron(
        sparse.eye(T), Bd
    )
    Aeq = sparse.hstack([Ax, Bu])

    A_ineq = sparse.vstack(
        [sparse.eye(T * nx + T * nu)]
    )

    

    A = sparse.vstack(
        [
            Aeq,
            A_ineq
        ]
    )

    # get b
    if np.isscalar(state_box):
        state_box_vec = state_box*np.ones(T * nx)
    else:
        state_box_vec = np.repeat(state_box, T)
    if np.isscalar(control_box):
        control_box_vec = control_box*np.ones(T * nu)
    else:
        control_box_vec = np.repeat(control_box, T)
    import pdb
    pdb.set_trace()
    b_upper = np.hstack(
        [state_box_vec, control_box_vec])
    b_lower = -np.hstack(
        [state_box_vec, control_box_vec])
    beq = np.zeros(T * nx)
    l = np.hstack([beq, b_lower])
    u = np.hstack([beq, b_upper])

    cones = dict(z=T * nx, l=2 * (T * nx + T * nu))

    out_dict = dict(cones=cones,
                    A=jnp.array(A.todense()),
                    P=jnp.array(P.todense()),
                    l=jnp.array(l),
                    u=jnp.array(u),
                    c=jnp.array(c),
                    A_dynamics=jnp.array(Ad))
    return out_dict


def static_canon(T, nx, nu, state_box, control_box, Q_val, QT_val, R_val, Ad=None, Bd=None):
    '''
    take in (nx, nu, )

    Q, R, q, QT, qT, xmin, xmax, umin, umax, T

    return (P, c, A, b) ... but b will change so not meaningful

    x0 is always the only thing that changes!
    (P, c, A) will be the same
    (b) will change in the location where x_init is!
    '''

    Q = Q_val * np.eye(nx)
    QT = QT_val * np.eye(nx)
    R = R_val * np.eye(nu)
    q = np.zeros(nx)  # np.random.normal(size=(nx))#
    qT = np.zeros(nx)

    if Ad is None and Bd is None:
        Ad = .1 * np.random.normal(size=(nx, nx))
        Bd = .1 * np.random.normal(size=(nx, nu))

    '''
    umin = xmin = -1
    umax = xmax = +1
    '''

    # Quadratic objective
    P_sparse = sparse.block_diag(
        [sparse.kron(sparse.eye(T-1), Q), QT, sparse.kron(sparse.eye(T), R)],
        format="csc",
    )

    # Linear objective
    c = np.hstack([np.kron(np.ones(T-1), q), qT, np.zeros(T * nu)])

    # Linear dynamics
    Ax = sparse.kron(sparse.eye(T + 1), -sparse.eye(nx)) + sparse.kron(
        sparse.eye(T + 1, k=-1), Ad
    )
    Ax = Ax[nx:, nx:]

    Bu = sparse.kron(
        sparse.eye(T), Bd
    )
    Aeq = sparse.hstack([Ax, Bu])

    beq = np.zeros(T * nx)
    # update the first nx entries of beq to be A@x_init

    '''
    top block for (x, u) <= (xmax, umax)
    bottom block for (x, u) >= (xmin, umin)
    i.e. (-x, -u) <= (-xmin, -umin)
    '''

    A_ineq = sparse.vstack(
        [sparse.eye(T * nx + T * nu),
         -sparse.eye(T * nx + T * nu)]
    )

    # Amid = sparse.csc_matrix((1, (T+1)*nx + T*nu))

    # stack A
    A_sparse = sparse.vstack(
        [
            Aeq,
            A_ineq
        ]
    )

    # get b
    b_upper = np.hstack(
        [state_box*np.ones(T * nx), control_box*np.ones(T * nu)])
    b_lower = np.hstack(
        [state_box*np.ones(T * nx), control_box*np.ones(T * nu)])
    b = np.hstack([beq, b_upper, b_lower])

    cones = dict(z=T * nx, l=2 * (T * nx + T * nu))
    # data = dict(P=P, c=c, A=A, b=b)
    cones_array = jnp.array([cones['z'], cones['l']])

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
    algo_factor = jsp.linalg.lu_factor(M + jnp.eye(n+m))

    A_sparse = csc_matrix(A)
    P_sparse = csc_matrix(P)

    out_dict = dict(M=M,
                    algo_factor=algo_factor,
                    cones_array=cones_array,
                    cones_dict=cones,
                    A_sparse=A_sparse,
                    P_sparse=P_sparse,
                    b=b,
                    c=c,
                    A_dynamics=Ad)

    return out_dict


def oscillating_masses_setup(nx, nu):
    dt = .5

    # spring constant
    c = 1

    # damping constant
    d = .1

    # other constant
    a, b = -2*c, -2

    # form Ac
    Ac = np.zeros((nx, nx))
    Ac[:18, 18:] = np.eye(18)
    L18 = np.eye(18, k=-1)  # + np.eye(18)

    bottom_left = a*np.eye(18) + c*L18 + c*L18.T
    bottom_right = b*np.eye(18) + d*L18 + d*L18.T
    Ac[18:, :18] = bottom_left
    Ac[18:, 18:] = bottom_right

    # form Bc
    F = np.zeros((6, 3))
    F[0, 0] = 1
    F[1, 0] = -1
    F[2, 1] = 1
    F[3, 1] = 1
    F[4, 2] = -1
    F[5, 2] = 1
    IF = np.kron(np.eye(3), F.T)

    Bc = np.hstack([
        np.zeros((nu, 18)),
        IF
    ]
    )
    Bc = Bc.T

    A = Ac * dt + np.eye(nx)
    B = Bc * dt

    return A, B


def multiple_random_osc_mass_osqp(N, T=50, x_init_box=2, state_box=4,
                                  control_box=.5, nx=36, nu=9, Q_val=1, QT_val=1, R_val=1,
                                  sigma=1, rho=1,
                                  seed=42):
    np.random.seed(seed)
    static_dict = static_canon_osqp(T, nx, nu, state_box, control_box, Q_val,
                                    QT_val, R_val, Ad=None, Bd=None)
    P, A = static_dict['P'], static_dict['A']
    c, l, u = static_dict['c'], static_dict['l'], static_dict['u']
    m, n = A.shape
    Ad = static_dict['A_dynamics']
    cones = static_dict['cones']

    q_mat = jnp.zeros((N, n + 2 * m))
    q_mat = q_mat.at[:, :n].set(c)

    # factor
    M = P + sigma * jnp.eye(n) + rho * A.T @ A
    factor = jsp.linalg.lu_factor(M)

    # x_init is theta
    x_init_mat = jnp.array(x_init_box * (2 * np.random.rand(N, nx) - 1))

    for i in range(N):
        # generate new rhs of first block constraint
        l = l.at[:nx].set(Ad @ x_init_mat[i, :])
        u = u.at[:nx].set(Ad @ x_init_mat[i, :])

        # convert to osqp
        # l, u = convert_scs_2_osqp(b_scs, cones)

        q_osqp = jnp.concatenate([c, l, u])
        qmat = q_mat.at[i, :].set(q_osqp)
    theta_mat = x_init_mat
    return factor, A, q_mat, theta_mat


def convert_scs_2_osqp(b_scs, cones):
    m = b_scs.size
    l, u = jnp.zeros(m), jnp.zeros(m)
    num_zeros = cones['z']
    num_ineq = cones['l']
    l = l.at[:num_zeros].set(b_scs[:num_zeros])
    l = l.at[num_zeros:].set(-jnp.inf)
    u = b_scs
    return l, u
