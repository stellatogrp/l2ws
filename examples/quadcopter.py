import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from scipy.linalg import solve_discrete_are
from examples.osc_mass import static_canon_osqp
import cvxpy as cp
import yaml
from l2ws.launcher import Workspace
import os
from examples.solve_script import osqp_setup_script
import matplotlib.pyplot as plt
from functools import partial
from jax import vmap
from scipy import sparse

QUADCOPTER_NX = 12
QUADCOPTER_NU = 4


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

    # setup the training
    T, x_init_factor = setup_cfg['T'], setup_cfg['x_init_factor']
    nx, nu = setup_cfg['nx'], setup_cfg['nu']
    quadcopter = setup_cfg.get('quadcopter', False)
    # num_traj = setup_cfg['num_traj']
    traj_length = setup_cfg['traj_length']
    mpc_setup = multiple_random_mpc_osqp(5,
                                         T=T,
                                         nx=nx,
                                         nu=nu,
                                         Ad=None,
                                         Bd=None,
                                         seed=setup_cfg['seed'],
                                         x_init_factor=x_init_factor,
                                         quadcopter=quadcopter)
    # factor, P, A, q_mat_train, theta_mat_train, x_bar, Ad, rho_vec = mpc_setup
    # factor, P, A, q_mat_train, theta_mat_train, x_bar, Ad, Bd, rho_vec = mpc_setup
    factor, P, A, q_mat_train, theta_mat_train, x_min, x_max, Ad, Bd, rho_vec = mpc_setup
    m, n = A.shape

    static_dict = dict(factor=factor, P=P, A=A, rho=rho_vec)

    # we directly save q now
    static_flag = True
    algo = 'osqp'

    partial_shifted_sol_fn = partial(shifted_sol, T=T, nx=nx, nu=nu, m=m, n=n)
    batch_shifted_sol_fn = vmap(partial_shifted_sol_fn, in_axes=(0), out_axes=(0))
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example,
                          traj_length=traj_length, shifted_sol_fn=batch_shifted_sol_fn)

    # run the workspace
    workspace.run()


def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test

    # np.random.seed(setup_cfg['seed'])
    # m_orig, n_orig = setup_cfg['m_orig'], setup_cfg['n_orig']
    # A = jnp.array(np.random.normal(size=(m_orig, n_orig)))
    # evals, evecs = jnp.linalg.eigh(A.T @ A)
    # ista_step = 1 / evals.max()
    # lambd = setup_cfg['lambd']

    np.random.seed(cfg.seed)

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    # P, A, cones, q_mat, theta_mat_jax, A_tensor = multiple_random_sparse_pca(
    #     n_orig, cfg.k, cfg.r, N, factor=False)
    # P_sparse, A_sparse = csc_matrix(P), csc_matrix(A)
    # b_mat = generate_b_mat(A, N, p=.1)
    # m, n = A.shape

    # setup the training
    T, x_init_factor = cfg.T, cfg.x_init_factor
    nx, nu = cfg.nx, cfg.nu
    noise_std_dev = cfg.noise_std_dev
    quadcopter = cfg.get('quadcopter', False)
    mpc_setup = multiple_random_mpc_osqp(N_train,
                                         T=T,
                                         nx=nx,
                                         nu=nu,
                                         Ad=None,
                                         Bd=None,
                                         seed=cfg.seed,
                                         x_init_factor=x_init_factor,
                                         quadcopter=quadcopter)
    factor, P, A, q_mat_train, theta_mat_train, x_min, x_max, Ad, Bd, rho_vec = mpc_setup

    # setup the testing
    q = q_mat_train[0, :]

    if quadcopter:
        num_traj_train = int(N_train / cfg.traj_length)
        theta_mat_train, z_stars_train, q_mat_train = solve_multiple_trajectories(
            cfg.traj_length, num_traj_train, x_min, x_max, x_init_factor, Ad, P, A, q)

        horizon = int(N_test / cfg.num_traj)

    num_traj_test = int(N_test / cfg.traj_length)
    theta_mat_test, z_stars_test, q_mat_test = solve_multiple_trajectories(
        cfg.traj_length, num_traj_test, x_min, x_max, x_init_factor, Ad, P, A, q, noise_std_dev)

    # create theta_mat and q_mat
    q_mat = jnp.vstack([q_mat_train, q_mat_test])
    theta_mat = jnp.vstack([theta_mat_train, theta_mat_test])
    # z_stars = jnp.vstack([z_stars_train, z_stars_test])

    # osqp_setup_script(theta_mat, q_mat, P, A, output_filename, z_stars=z_stars)
    osqp_setup_script(theta_mat, q_mat, P, A, output_filename, z_stars=None)
    # import pdb
    # pdb.set_trace()


def shifted_sol(z_star, T, nx, nu, m, n):
    # shifted_z_star = jnp.zeros(z_star.size)

    x_star = z_star[:n]
    y_star = z_star[n:]

    shifted_x_star = jnp.zeros(n)
    shifted_y_star = jnp.zeros(m)

    # indices markers
    end_state = nx * T
    end_dyn_cons = nx * T
    end_state_cons = 2 * nx * T

    # get primal vars
    shifted_states = x_star[nx:end_state]
    shifted_controls = x_star[end_state + nu:]

    # insert into shifted x_star
    shifted_x_star = shifted_x_star.at[:end_state - nx].set(shifted_states)
    shifted_x_star = shifted_x_star.at[end_state:-nu].set(shifted_controls)

    # get dual vars
    shifted_dyn_cons = y_star[nx:end_dyn_cons]
    shifted_state_cons = y_star[end_dyn_cons + nx:end_state_cons]
    shifted_control_cons = y_star[end_state_cons + nu:]

    # insert into shifted y_star
    shifted_y_star = shifted_y_star.at[:end_dyn_cons - nx].set(shifted_dyn_cons)
    shifted_y_star = shifted_y_star.at[end_dyn_cons:end_state_cons - nx].set(shifted_state_cons)
    shifted_y_star = shifted_y_star.at[end_state_cons:-nu].set(shifted_control_cons)

    # concatentate primal and dual
    shifted_z_star = jnp.concatenate([shifted_x_star, shifted_y_star])

    return shifted_z_star


def generate_static_prob_data(nx, nu, seed):
    np.random.seed(seed)
    x_bar = 1 + np.random.rand(nx)
    u_bar = .1 * np.random.rand(nu)

    dA = .1 * np.random.normal(size=(nx, nx))
    orig_Ad = np.eye(nx) + dA

    # normalize Ad so eigenvalues are less than 1
    orig_evals, evecs = np.linalg.eig(orig_Ad)
    max_norm = np.max(np.abs(orig_evals))
    if max_norm >= 1:
        Ad = orig_Ad / max_norm
    else:
        Ad = orig_Ad

    Bd = np.random.normal(size=(nx, nu))

    # create cost matrices
    R = .1 * np.eye(nu)

    q_vec = np.random.rand(nx) * 10
    p = 0.7  # 1.0
    q_vec_mask = np.random.choice([0, 1], size=(nx), p=[1-p, p], replace=True)
    q_vec = np.multiply(q_vec, q_vec_mask)
    Q = np.diag(q_vec)
    QT = solve_discrete_are(Ad, Bd, Q, R)

    # return Ad, Bd, Q, QT, R, x_bar, u_bar
    x_ref = np.zeros(nx)
    x_max = x_bar
    x_min = -x_bar
    u_max = u_bar
    u_min = -u_bar
    return Ad, Bd, Q, QT, R, x_ref, x_min, x_max, u_min, u_max


def generate_static_prob_data_quadcopter():
    # dynamics
    Ad = np.array([
        [1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.],
        [0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.],
        [0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.],
        [0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.],
        [0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.],
        [0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992],
        [0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.],
        [0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.],
        [0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.],
        [0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.],
        [0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.],
        [0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846]
    ])
    Bd = np.array([
        [0.,      -0.0726,  0.,     0.0726],
        [-0.0726,  0.,      0.0726, 0.],
        [-0.0152,  0.0152, -0.0152, 0.0152],
        [-0.,     -0.0006, -0.,     0.0006],
        [0.0006,   0.,     -0.0006, 0.0000],
        [0.0106,   0.0106,  0.0106, 0.0106],
        [0,       -1.4512,  0.,     1.4512],
        [-1.4512,  0.,      1.4512, 0.],
        [-0.3049,  0.3049, -0.3049, 0.3049],
        [-0.,     -0.0236,  0.,     0.0236],
        [0.0236,   0.,     -0.0236, 0.],
        [0.2107,   0.2107,  0.2107, 0.2107]
    ])
    [nx, nu] = Bd.shape

    # Objective function
    # Q = np.array(sparse.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.]))
    # Q = np.diag([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
    # Q = np.diag([10., 10., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
    Q = np.diag([10., 10., 10., 10., 10., 10., 10., 10., 10., 5., 5., 5.]) / 10
    # Q = np.diag([.1, .1, 10., 10., 10., 10., .1, .1, .1, 5., 5., 5.])
    QT = Q
    R = .1 * np.eye(4)  # 0.1 * np.eye(4)

    # - linear objective
    # q = np.hstack([np.kron(np.ones(N), -Q@xr), -QN@xr, np.zeros(N*nu)])
    x_ref = np.array([.5, .5, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    # Constraints
    u0 = 10.5916
    u_min = np.array([9.6, 9.6, 9.6, 9.6]) - u0
    u_max = np.array([13., 13., 13., 13.]) - u0
    x_bds = 100
    x_min = np.array([-x_bds, -x_bds, -np.inf, -np.inf, -np.inf, -1.,
                      -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    x_max = np.array([x_bds, x_bds, np.inf, np.inf, np.inf, np.inf,
                      np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    # x_min = np.array([-np.pi/6,-np.pi/6,-np.inf,-np.inf,-np.inf,-1.,
    #                 -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf])
    # x_max = np.array([ np.pi/6, np.pi/6, np.inf, np.inf, np.inf, np.inf,
    #                 np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

    return Ad, Bd, Q, QT, R, x_ref, x_min, x_max, u_min, u_max


def multiple_random_mpc_osqp(N,
                             T=10,
                             x_init_factor=.3,
                             nx=20,
                             nu=10,
                             sigma=1,
                             rho=1,
                             Ad=None,
                             Bd=None,
                             seed=42,
                             quadcopter=False):
    if quadcopter:
        Ad, Bd, Q, QT, R, x_ref, x_min, x_max, u_min, u_max = generate_static_prob_data_quadcopter()
        nx, nu = Bd.shape
        # x_init_mat = .1 * (2 * np.random.rand(N, nx) - 1)
        # x_init_mat = .4 * (2 * np.random.rand(N, nx) - 1)
        x_init_mat = x_init_factor * (2 * np.random.rand(N, nx) - 1)
        # x_init_mat[:, 2] = 2 * (2 * np.random.rand(N) - 1)
        x_init_mat[:, 3:] = 0
        # x_init_mat[:, 1] = 0
    else:
        Ad, Bd, Q, QT, R, x_ref, x_min, x_max, u_min, u_max = generate_static_prob_data(
            nx, nu, seed)
    # static_dict = static_canon_osqp(T, nx, nu, x_bar, u_bar, Q,
    #                                 QT, R, Ad=Ad, Bd=Bd)
    static_dict = static_canon_osqp(T, nx, nu, x_min, x_max, u_min, u_max, Q,
                                    QT, R, x_ref, Ad=Ad, Bd=Bd)
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
    # x_init_mat = x_init_factor * jnp.array(x_bar * (2 * np.random.rand(N, nx) - 1))
    if not quadcopter:
        x_diff = jnp.array(x_max - x_min)
        x_center = x_min + x_diff / 2
        x_init_mat = x_center + x_init_factor * (x_diff / 2) * (2 * np.random.rand(N, nx) - 1)
    # import pdb
    # pdb.set_trace()

    for i in range(N):
        # generate new rhs of first block constraint
        l = l.at[:nx].set(-Ad @ x_init_mat[i, :])
        u = u.at[:nx].set(-Ad @ x_init_mat[i, :])

        q_osqp = jnp.concatenate([c, l, u])
        q_mat = q_mat.at[i, :].set(q_osqp)
    theta_mat = x_init_mat
    # return factor, P, A, q_mat, theta_mat
    # import pdb
    # pdb.set_trace()

    return factor, P, A, q_mat, theta_mat, x_min, x_max, Ad, Bd, rho_vec


def solve_many_probs_cvxpy(P, A, q_mat):
    """
    solves many QPs where each problem has a different b vector
    """
    # q_mat_finite = q_mat
    q_mat = q_mat.at[q_mat == np.inf].set(10000)
    q_mat = q_mat.at[q_mat == -np.inf].set(-10000)
    # import pdb
    # pdb.set_trace()
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


def solve_multiple_trajectories(traj_length, num_traj, x_min, x_max, x_init_factor, Ad, P, A, q, noise_std_dev):
    m, n = A.shape
    nx = Ad.shape[0]
    q_mat = jnp.zeros((traj_length * num_traj, n + 2 * m))

    # old
    # first_x_inits = x_init_factor * jnp.array(x_bar * (2 * np.random.rand(num_traj, nx) - 1))

    # new
    if nx != 12:
        x_diff = jnp.array(x_max - x_min)
        x_center = x_min + x_diff / 2
        first_x_inits = x_center + x_init_factor * \
            (x_diff / 2) * (2 * np.random.rand(num_traj, nx) - 1)
    else:
        # first_x_inits = 0.1 * (2 * np.random.rand(num_traj, nx) - 1)
        # first_x_inits = .4 * (2 * np.random.rand(num_traj, nx) - 1)
        first_x_inits = x_init_factor * (2 * np.random.rand(num_traj, nx) - 1)
        # first_x_inits[:, 2] = 2 * (2 * np.random.rand(num_traj) - 1)
        first_x_inits[:, 3:] = 0
        # first_x_inits[:, 1] = 0

    theta_mat_list = []
    z_stars_list = []
    q_mat_list = []
    for i in range(num_traj):
        first_x_init = first_x_inits[i, :]
        theta_mat_curr, z_stars_curr, q_mat_curr = solve_trajectory(first_x_init, P, A, q,
                                                                    traj_length, Ad, noise_std_dev)
        theta_mat_list.append(theta_mat_curr)
        z_stars_list.append(z_stars_curr)
        q_mat_list.append(q_mat_curr)
    theta_mat = jnp.vstack(theta_mat_list)
    z_stars = jnp.vstack(z_stars_list)
    q_mat = jnp.vstack(q_mat_list)
    return theta_mat, z_stars, q_mat


def solve_trajectory(first_x_init, P_orig, A, q, traj_length, Ad, noise_std_dev):
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
    P = cp.atoms.affine.wraps.psd_wrap(P_orig)
    m, n = A.shape
    x, w = cp.Variable(n), cp.Variable(m)
    c_param, l_param, u_param = cp.Parameter(n), cp.Parameter(m), cp.Parameter(m)
    constraints = [A @ x == w, l_param <= w, w <= u_param]
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, P) + c_param @ x), constraints)

    c_param.value = np.array(q[:n])

    # z_stars = jnp.zeros((traj_length, m + n))
    z_stars = jnp.zeros((traj_length, 2 * m + n))
    theta_mat = jnp.zeros((traj_length, nx))
    q_mat = jnp.zeros((traj_length, n + 2 * m))
    q_mat = q_mat.at[:, :].set(q)

    # set the first x_init
    x_init = first_x_init

    for i in range(traj_length):
        l = q[n:n + m]
        u = q[n + m:]
        theta_mat = theta_mat.at[i, :].set(x_init)
        Ad_x_init = Ad @ x_init
        l = l.at[:nx].set(-Ad_x_init)
        u = u.at[:nx].set(-Ad_x_init)
        l_np = np.array(l)
        u_np = np.array(u)
        l_np[l_np == -np.inf] = -10000
        u_np[u_np == np.inf] = 10000
        # import pdb
        # pdb.set_trace()
        l_param.value = l_np
        u_param.value = u_np
        print('i', i)
        prob.solve(verbose=False)

        x_star = jnp.array(x.value)
        y_star = jnp.array(constraints[0].dual_value)
        w_star = jnp.array(w.value)

        z_star = jnp.concatenate([x_star, y_star, w_star])
        # print('z_star', z_star[:20])
        z_stars = z_stars.at[i, :].set(z_star)

        q_mat = q_mat.at[i, n:n + m].set(l)
        q_mat = q_mat.at[i, n + m:].set(u)

        # set the next x_init
        # x_init = x_star[nx:2 * nx]
        noise = noise_std_dev * jnp.array(np.random.normal(size=(nx,)))  # * x_bar
        x_init = x_star[:nx] + noise
        # print('x_init', x_init)
    return theta_mat, z_stars, q_mat


# def quadcopter_dynamics(state, inputs, t):
#     # State variables
#     position = state[:3]  # Position [x, y, z]
#     velocity = state[3:6]  # Velocity [vx, vy, vz]
#     quaternion = state[6:]  # Angular velocity of the body frame [q_w, q_x, q_y, q_z]
#     qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

#     # inputs is u = [thrust, wx, wy, wz]

#     # 0.5 * ( -wx*qx - wy*qy - wz*qz ),
#     # 0.5 * (  wx*qw + wz*qy - wy*qz ),
#     # 0.5 * (  wy*qw - wz*qx + wx*qz ),
#     # 0.5 * (  wz*qw + wy*qx - wx*qy ),
#     # 2 * (qw*qy + qx*qz) * thrust,
#     # 2 * (qy*qz - qw*qx) * thrust,
#     # (qw*qw - qx*qx - qy*qy + qz*qz) * thrust - self._gz
#     # (1 - 2*qx*qx - 2*qy*qy) * thrust - self._gz

#     position_dot = velocity
#     quaternion_dot = quaternion_product(inputs[1:], quaternion)

#     thrust = inputs[0]

#     velocity_dot = jnp.array([2 * (qw*qy + qx*qz) * thrust,
#                                 2 * (qy*qz - qw*qx) * thrust,
#                                 (qw*qw - qx*qx - qy*qy + qz*qz) * thrust - 9.8])
#     state_dot = jnp.concatenate([position_dot, velocity_dot, quaternion_dot])

#     return state_dot


def quadcopter_dynamics(state, thrusts, t):
    # State variables
    position = state[:3]  # Position [x, y, z]
    velocity = state[3:6]  # Velocity [vx, vy, vz]
    theta = state[6:9]  # Angles of the inertial frame (roll, pitch, yaw) [r, p, y]
    omega = state[9:]  # Angular velocity of the body frame [w_x, w_y, w_z]

    # constants
    Ixx, Iyy, Izz = 1, 1, 1
    I = jnp.array([
        [Ixx, 0, 0],
        [0, Iyy, 0],
        [0, 0, Izz]
    ])
    I_inv = jnp.array([
        [1 / Ixx, 0, 0],
        [0, 1 / Iyy, 0],
        [0, 0, 1 / Izz]
    ])
    gravity = jnp.array([0, 0, -9.8])
    mass = 1
    k_drag = 1
    k_thrust = 10
    k_torque = 100  # 1
    k = 1
    b = 1
    L = 1

    # rotation matrix
    R = get_rotation_matrix(theta)

    # calculate forces and torques
    drag_force = -k_drag * velocity
    # thrust_force = k_thrust * R @ jnp.array([thrusts[3] + thrusts[1] - thrusts[2] - thrusts[0],
    #                                          thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3],
    #                                          k * jnp.sum(thrusts[:4])])
    # thrust_x = thrusts[1] + thrusts[3] - thrusts[0] - thrusts[2]  # Thrust difference between right and left propellers
    # thrust_y = thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3]
    # thrust_force = k_thrust * R @ jnp.array([ 0,
    #                                           0,
    #                                          k * jnp.sum(thrusts[:4])])
    thrust_force = k_thrust * R @ jnp.array([0, 0, k * jnp.sum(thrusts[:4])])
    # print('thrust_force', thrust_force)

    # horizontal_force = jnp.array([thrusts[4], thrusts[5], 0])
    torques = k_torque * jnp.array([L * k * (thrusts[0] - thrusts[2]),
                                    L * k * (thrusts[1] - thrusts[3]),
                                    b * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3])])
    # print('torque', torques)

    position_dot = velocity
    theta_dot = omega_2_thetadot(theta, omega)
    velocity_dot = gravity + (thrust_force + drag_force) / mass
    omega_dot = I_inv @ (torques - jnp.cross(omega, I @ omega))
    state_dot = jnp.concatenate([position_dot, velocity_dot, theta_dot, omega_dot])

    return state_dot


def get_rotation_matrix(theta):
    roll, pitch, yaw = theta[0], theta[1], theta[2]
    s_roll = jnp.sin(roll)
    c_roll = jnp.cos(roll)
    s_pitch = jnp.sin(pitch)
    c_pitch = jnp.cos(pitch)
    s_yaw = jnp.sin(yaw)
    c_yaw = jnp.cos(yaw)
    R = jnp.array([[c_roll * c_yaw - c_pitch * s_roll * s_yaw, -c_yaw * s_roll - c_roll * c_pitch * s_yaw, s_pitch * s_yaw],
                   [c_pitch * c_yaw * s_yaw + c_roll * s_yaw, c_roll *
                       c_pitch * c_yaw - s_roll * s_yaw, -c_yaw * s_pitch],
                   [s_pitch * s_roll, c_roll * s_pitch, c_pitch]])
    # print('theta', theta)
    # print('R', R)
    return R


def omega_2_thetadot(theta, omega):
    """
    given theta and theta_dot, the roll, pitch, yaw of the inertial frame (and its derivative)
    returns omega, the angular velocity of the body frame

    omega = A(theta) theta_dot
    theta_dot = A(theta)^{-1} omega
    """
    roll, pitch, yaw = theta[0], theta[1], theta[2]
    s_roll = jnp.sin(roll)
    c_roll = jnp.cos(roll)
    s_pitch = jnp.sin(pitch)
    c_pitch = jnp.cos(pitch)
    A = jnp.array([[1, 0, -s_pitch],
                   [0, c_roll, c_pitch * s_roll],
                   [0, -s_roll, c_pitch * c_roll]])
    theta_dot = jnp.linalg.inv(A) @ omega
    return theta_dot


def thetadot_2_omega(theta, theta_dot):
    """
    given theta and theta_dot, the roll, pitch, yaw of the inertial frame (and its derivative)
    returns omega, the angular velocity of the body frame

    omega = A(theta) theta_dot
    theta_dot = A(theta)^{-1} omega
    """
    roll, pitch, yaw = theta[0], theta[1], theta[2]
    s_roll = jnp.sin(roll)
    c_roll = jnp.cos(roll)
    s_pitch = jnp.sin(pitch)
    c_pitch = jnp.cos(pitch)
    A = jnp.array([[1, 0, -s_pitch],
                   [0, c_roll, c_pitch * s_roll],
                   [0, -s_roll, c_pitch * s_roll]])
    omega = A @ theta_dot
    return omega


# def quadcopter_dynamics(state, thrusts, t):
#     """
#     x = state
#     u = thrusts
#     returns x_dot = f(x, u)

#     written in jax so we can use autodifferentiation to compute the linearized dynamics
#     """
#     # State variables
#     position = state[:3]  # Position [x, y, z]
#     velocity = state[3:6]  # Velocity [vx, vy, vz]
#     quaternion = state[6:10]  # Quaternion [qw, qx, qy, qz]
#     angular_velocity = state[10:]  # Angular velocity [p, q, r]

#     # Constants
#     mass = 1.0  # Quadcopter mass
#     g = 9.81  # Acceleration due to gravity

#     # Thrust and torque constants
#     k_thrust = 5  # Thrust coefficient
#     k_torque = 1 #0.25  # Torque coefficient

#     # Calculate forces and torques
#     forces = jnp.array([0.0, 0.0, -mass * g])  # Gravity force
#     # thrusts = jnp.clip(thrusts, 0.0, jnp.inf)  # Ensure thrusts are non-negative
#     # body_z = quaternion_to_rotation_matrix(quaternion)[:, 2]  # Body z-axis in world frame
#     body_z = jnp.array([
#         0*velocity[0]**2-2 * (quaternion[0] * quaternion[2] + quaternion[1] * quaternion[3]),
#         0*velocity[1]**2-2 * (-quaternion[0] * quaternion[1] + quaternion[2] * quaternion[3]),
#         0*velocity[2]**2-2 * (quaternion[0] ** 2 - quaternion[1] ** 2 - quaternion[2] ** 2 + quaternion[3] ** 2),
#     ]) / mass
#     print('body_z', body_z)
#     forces += k_thrust * jnp.sum(thrusts) * body_z  # Thrust forces
#     # print('forces', forces)

#     torques = jnp.array([
#         k_torque * (thrusts[1] - thrusts[3]),  # Roll torque
#         k_torque * (thrusts[2] - thrusts[0]),  # Pitch torque
#         k_torque * (thrusts[1] + thrusts[3] - thrusts[0] - thrusts[2])  # Yaw torque
#     ])

#     # Calculate state derivatives

#     # Position derivative
#     position_dot = velocity

#     # Velocity derivative
#     velocity_dot = forces / mass

#     # Quaternion derivative
#     # quaternion_dot = 0.5 * quaternion_product(quaternion,
#     #                                           jnp.concatenate([jnp.array([0]), angular_velocity]))
#     quaternion_dot = quaternion_product(angular_velocity, quaternion)

#     # Angular velocity derivative
#     # inertia_matrix_inv = inertia_matrix_inverse(quaternion)

#     Ixx = 1e-2
#     Iyy = 1e-2
#     Izz = 1e-2

#     I = jnp.array([
#         [Ixx, 0, 0],
#         [0, Iyy, 0],
#         [0, 0, Izz]
#     ])

#     I_inv = jnp.array([
#         [1 / Ixx, 0, 0],
#         [0, 1 / Iyy, 0],
#         [0, 0, 1 / Izz]
#     ])
#     # print('torque', torques)

#     angular_velocity_dot = I_inv @ (torques - jnp.cross(angular_velocity, I @ angular_velocity))
#     # inertia_matrix_inv.dot(torques) - jnp.cross(angular_velocity, angular_velocity)
#     # angular_velocity_dot =

#     # Concatenate the state derivatives
#     # state_dot = jnp.concatenate([position_dot, velocity_dot, quaternion_dot[1:], angular_velocity_dot])
#     state_dot = jnp.concatenate([position_dot, velocity_dot, quaternion_dot, angular_velocity_dot])

#     return state_dot


def YPRToQuat(rpy):
    # For ZYX, Yaw-Pitch-Roll
    # psi   = RPY[0] = r1
    # theta = RPY[1] = r2
    # phi   = RPY[2] = r3
    r1, r2, r3 = rpy[0], rpy[1], rpy[2]

    cr1 = jnp.cos(0.5*r1)
    cr2 = jnp.cos(0.5*r2)
    cr3 = jnp.cos(0.5*r3)
    sr1 = jnp.sin(0.5*r1)
    sr2 = jnp.sin(0.5*r2)
    sr3 = jnp.sin(0.5*r3)

    q0 = cr1*cr2*cr3 + sr1*sr2*sr3
    q1 = cr1*cr2*sr3 - sr1*sr2*cr3
    q2 = cr1*sr2*cr3 + sr1*cr2*sr3
    q3 = sr1*cr2*cr3 - cr1*sr2*sr3

    # e0,e1,e2,e3 = qw,qx,qy,qz
    q = jnp.array([q0, q1, q2, q3])
    # q = q*np.sign(e0)

    q = q / jnp.linalg.norm(q)

    return q


def quaternion_to_rotation_matrix(quaternion):
    w, x, y, z = quaternion
    return jnp.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 1 - 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 1 - 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]
    ])


# def quaternion_product(q1, q2):
    # w1, x1, y1, z1 = q1
    # w2, x2, y2, z2 = q2
    # w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    # x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    # y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    # z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    # return jnp.array([w, x, y, z])


def quaternion_product(p, q):
    p0, p1, p2 = p
    q0, q1, q2, q3 = q
    w = -0.5*p0*q1 - 0.5*p1*q2 - 0.5*q3*p2
    x = 0.5*p0*q0 - 0.5*p1*q3 + 0.5*q2*p2
    y = 0.5*p0*q3 + 0.5*p1*q0 - 0.5*q1*p2
    z = -0.5*p0*q2 + 0.5*p1*q1 + 0.5*q0*p2
    return jnp.array([w, x, y, z])


def inertia_matrix_inverse(quaternion):
    # Assuming diagonal inertia matrix
    Ixx = .01  # 1.0
    Iyy = .01  # 1.0
    Izz = .02  # 1.0

    qw, qx, qy, qz = quaternion

    I_inv = jnp.array([
        [1 / Ixx, 0, 0],
        [0, 1 / Iyy, 0],
        [0, 0, 1 / Izz]
    ])

    R = quaternion_to_rotation_matrix(quaternion)

    return R @ I_inv @ R.T


def plot_traj_3d(state_traj_list, goals, labels):
    """
    state_traj_list is a list of lists
    """
    # Quadcopter dimensions
    body_length = .1
    body_width = .1
    body_height = .01

    # Body coordinates
    body_coords = np.array([
        [-body_length/2, body_length/2, body_length/2, -body_length/2, -body_length/2],
        [-body_width/2, -body_width/2, body_width/2, body_width/2, -body_width/2],
        [0, 0, 0, 0, 0]
    ])

    # Propeller coordinates in the body-fixed frame
    prop_coords = np.array([
        [-body_length / 2, -body_length / 2, body_length / 2, body_length / 2],
        [body_width / 2, -body_width / 2, -body_width / 2, body_width / 2],
        [0, 0, 0, 0]
    ])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for j in range(len(labels)):
        curr_state_traj = state_traj_list[j]
        xs = np.array([curr_state_traj[i][0] for i in range(len(curr_state_traj))])
        ys = np.array([curr_state_traj[i][1] for i in range(len(curr_state_traj))])
        zs = np.array([curr_state_traj[i][2] for i in range(len(curr_state_traj))])
        rolls = np.array([curr_state_traj[i][6] for i in range(len(curr_state_traj))])
        pitchs = np.array([curr_state_traj[i][7] for i in range(len(curr_state_traj))])
        yaws = np.array([curr_state_traj[i][8] for i in range(len(curr_state_traj))])

        for i in range(len(rolls)):
            roll = rolls[i]
            pitch = pitchs[i]
            yaw = yaws[i]

            # Rotate body coordinates
            R_roll = np.array([[1, 0, 0],
                               [0, np.cos(roll), -np.sin(roll)],
                               [0, np.sin(roll), np.cos(roll)]])
            R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                                [0, 1, 0],
                                [-np.sin(pitch), 0, np.cos(pitch)]])
            R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                              [np.sin(yaw), np.cos(yaw), 0],
                              [0, 0, 1]])

            body_coords_rotated = R_yaw @ R_pitch @ R_roll @ body_coords
            prop_coords_rotated = R_yaw @ R_pitch @ R_roll @ prop_coords
            # import pdb
            # pdb.set_trace()

            # Plot body
            # ax.plot3D(xs[i] + body_coords_rotated[0], ys[i] +
            #           body_coords_rotated[1], zs[i] + body_coords_rotated[2], 'b')
            
            # import pdb
            # pdb.set_trace()

            # Plot propellers
            # for k in range(4):
            #     ax.scatter(xs[i] + prop_coords_rotated[0, k], ys[i] + prop_coords_rotated[1, k], zs[i] + prop_coords_rotated[2, k], c='r')
            # propeller_positions = 
            propeller_positions = prop_coords_rotated.T + np.array([xs[i], ys[i], zs[i]])

            # propellers 0 and 1 (to center) are red

            # Plot the cross-shaped body of the quadcopter
            # body_x = xs[i] + np.array([propeller_positions[0, 0], propeller_positions[2, 0]])
            # body_y = ys[i] + np.array([propeller_positions[0, 1], propeller_positions[2, 1]])
            # body_z = zs[i] + np.array([propeller_positions[0, 2], propeller_positions[2, 2]])
            # ax.plot(body_x, body_y, body_z, 'b')

            # propeller 0 to center
            body_x = np.array([propeller_positions[0, 0], xs[i]])
            body_y = np.array([propeller_positions[0, 1], ys[i]])
            body_z = np.array([propeller_positions[0, 2], zs[i]])
            ax.plot(body_x, body_y, body_z, 'b')

            # propeller 1 to center
            body_x = np.array([propeller_positions[1, 0], xs[i]])
            body_y = np.array([propeller_positions[1, 1], ys[i]])
            body_z = np.array([propeller_positions[1, 2], zs[i]])
            ax.plot(body_x, body_y, body_z, 'b')

            # propeller 2 to center
            body_x = np.array([propeller_positions[2, 0], xs[i]])
            body_y = np.array([propeller_positions[2, 1], ys[i]])
            body_z = np.array([propeller_positions[2, 2], zs[i]])
            ax.plot(body_x, body_y, body_z, 'r')

            # propeller 3 to center
            body_x = np.array([propeller_positions[3, 0], xs[i]])
            body_y = np.array([propeller_positions[3, 1], ys[i]])
            body_z = np.array([propeller_positions[3, 2], zs[i]])
            ax.plot(body_x, body_y, body_z, 'r')

            # body_x = xs[i] + np.array([propeller_positions[1, 0], propeller_positions[3, 0]])
            # body_y = ys[i] + np.array([propeller_positions[1, 1], propeller_positions[3, 1]])
            # body_z = zs[i] + np.array([propeller_positions[1, 2], propeller_positions[3, 2]])
            # ax.plot(body_x, body_y, body_z, 'b')
            # body_x = np.array([propeller_positions[1, 0], propeller_positions[3, 0]])
            # body_y = np.array([propeller_positions[1, 1], propeller_positions[3, 1]])
            # body_z = np.array([propeller_positions[1, 2], propeller_positions[3, 2]])
            # ax.plot(body_x, body_y, body_z, 'b')

        # if labels[j] == 'optimal':
        #     ax.scatter(x, y, z, label=labels[j], color='green')
        # else:
        #     ax.scatter(x, y, z, label=labels[j])

        # plot the goals
        weights = np.arange(1, len(goals) + 1) / (len(goals) + 1)
        for i in range(len(goals)):
            x, y, z = goals[i][0], goals[i][1], goals[i][2]
            # ax.scatter(x, y, z, cmap='Reds_r', c=weights[i], label=f"goal {i}")
            ax.scatter(x, y, z, label=f"goal {i}")
    plt.legend()
    plt.show()


# def makeWaypoints():
#     deg2rad = jnp.pi / 180.0

#     v_average = 1.6

#     t_ini = 3
#     t = jnp.array([2, 0, 2, 0])

#     wp_ini = jnp.array([0, 0, 0])
#     wp = jnp.array([[2, 2, 1],
#                    [-2, 3, -3],
#                    [-2, -1, -3],
#                    [3, -2, 1],
#                    wp_ini]) / 10

#     yaw_ini = 0
#     yaw = jnp.array([20, -90, 120, 45])

#     t = jnp.hstack((t_ini, t)).astype(float)
#     # wp = jnp.vstack((wp_ini, wp)).astype(float)
#     yaw = jnp.hstack((yaw_ini, yaw)).astype(float) * deg2rad

#     return t, wp, yaw, v_average


def make_obstacle_course():
    deg2rad = jnp.pi / 180.0
    goals = jnp.array([[2, 2, 1],
                       [-2, 3, -3],
                       [-2, -1, -3],
                       [3, -2, 1],
                       [0, 0, 0]]) # / 10
    # goals = jnp.array([[1, 1, 1],
    #                    [-2, 3, -3],
    #                    [-2, -1, -3],
    #                    [3, -2, 1],
    #                    [0, 0, 0]]) / 10
    # goals_np = .1 * np.random.rand(5, 3) #.2 * np.random.rand(5, 3) - .1
    # goals = jnp.array(goals_np)
    rpys = 0  # * jnp.array([[.2, .2, 0],
    #    [.2, .2, 0],
    #    [.2, .2, 0],
    #    [.2, .2, 0],
    #    [0, 0, 0]])
    yaw_ini = 0
    yaw = np.array([20, -90, 120, 45])

    # t = np.hstack((t_ini, t)).astype(float)
    # wp = np.vstack((wp_ini, wp)).astype(float)
    yaw = np.hstack((yaw, yaw_ini)).astype(float)*deg2rad
    nx = QUADCOPTER_NX
    traj_list = []
    for i in range(5):
        ref = jnp.zeros(nx)
        # import pdb
        # pdb.set_trace()
        ref = ref.at[:3].set(goals[i, :])
        # ref = ref.at[6:9].set(rpys[i, :])
        # ref = ref.at[8].set(yaw[i])
        traj_list.append(ref)

    return traj_list
