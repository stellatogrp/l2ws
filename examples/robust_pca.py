import functools
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
from jax import vmap
import pandas as pd
from utils.generic_utils import vec_symm, unvec_symm
from scipy.spatial import distance_matrix


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
    }
)
log = logging.getLogger(__name__)


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
    example = "robust_pca"
    data_yaml_filename = f"{orig_cwd}/outputs/{example}/aggregate_outputs/{datetime}/data_setup_copied.yaml"

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    static_dict = static_canon(
        setup_cfg['p'],
        setup_cfg['q']
    )
    A_sparse = static_dict['A_sparse']
    m, n = A_sparse.shape

    get_q_single = functools.partial(single_q,
                                     m=m,
                                     n=n,
                                     p=setup_cfg['p'],
                                     q=setup_cfg['q'])
    get_q = vmap(get_q_single, in_axes=0, out_axes=0)

    """
    static_flag = True
    means that the matrices don't change across problems
    we only need to factor once
    """
    static_flag = True

    '''
    low_2_high_dim
    '''
    cones = static_dict['cones_dict']
    x_psd_size = cones['s'][0] # TOFIX in general
    y_psd_size = x_psd_size
    full_psd_size = int(x_psd_size * (x_psd_size + 1) / 2)
    n_x_non_psd = n - full_psd_size
    n_y_non_psd = m - int(y_psd_size * (y_psd_size + 1) / 2)
    n_x_low = n_x_non_psd + run_cfg.dx * x_psd_size
    n_y_low = n_y_non_psd + run_cfg.dy * y_psd_size

    # extract the right part of x_psd
    A_dense = A_sparse.todense()
    A_psd = A_dense[-full_psd_size:, :]
    where_out = np.where(A_psd < 0)
    x_psd_indices = where_out[1]

    y_psd_indices = n_y_non_psd + jnp.arange(int(y_psd_size * (y_psd_size + 1) / 2))     

    low_2_high_dim = functools.partial(low_2_high_dim_prediction,
                                       n_x_low=n_x_low,
                                       n_y_low=n_y_low,
                                       n_x_non_psd=n_x_non_psd,
                                       n_y_non_psd=n_y_non_psd,
                                       dx=run_cfg.dx,
                                       dy=run_cfg.dy,
                                       x_psd_size=x_psd_size,
                                       y_psd_size=y_psd_size,
                                       tx=run_cfg.tx,
                                       ty=run_cfg.ty,
                                       x_psd_indices=x_psd_indices
                                       )
    # in_axes_x = [None for i in range(run_cfg.tx)]
    # in_axes_y = [None for i in range(run_cfg.ty)]
    # low_2_high_dim_batch = vmap(low_2_high_dim, in_axes=(0, in_axes_x, in_axes_y), out_axes=0)

    workspace = Workspace(run_cfg, static_flag, static_dict, example,
                          get_q, low_2_high_dim=low_2_high_dim,
                          x_psd_indices=x_psd_indices, y_psd_indices=y_psd_indices)

    """
    run the workspace
    """
    workspace.run()


def sample_theta(p, q, sparse_frac, low_rank, A_star=None, B_star=None):
    # L, S, M are (p, q)
    # A is (p, r), B is (q, r)
    if A_star is None:
        A_star = np.random.normal(size=(p, low_rank))
    if B_star is None:
        B_star = np.random.normal(size=(q, low_rank))
    else:
        B_star = B_star + .1 * np.random.rand(q, low_rank)
    L_star = A_star @ B_star.T

    # generate random, sparse S_star
    S_original = np.random.normal(size=(p, q))
    S_mask = np.random.choice(2, size=(p, q), replace=True,
                              p=np.array([1 - sparse_frac, sparse_frac]))
    S_star = np.multiply(S_original, S_mask)

    M = L_star + S_star
    M_vec = np.ravel(M)
    mu_star = np.sum(np.abs(S_star))
    theta = np.append(mu_star, M_vec)
    return theta


def setup_probs(setup_cfg):
    print("entered robust kalman setup", flush=True)
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    p, q = cfg.p, cfg.q

    '''
    sample theta
    '''
    thetas_np = np.zeros((N, 1 + p * q))
    if cfg.A_star_seed is not None:
        np.random.seed(cfg.A_star_seed)
        A_star = np.random.normal(size=(p, cfg.low_rank))
    else:
        A_star = None
    B_star = np.random.rand(q, cfg.low_rank)
    for i in range(N):
        thetas_np[i, :] = sample_theta(p, q, cfg.sparse_frac, cfg.low_rank, A_star=A_star, B_star=B_star)
    thetas = jnp.array(thetas_np)

    """
    - canonicalize according to whether we have states or not
    - extract information dependent on the setup
    """
    log.info("creating static canonicalization...")
    t0 = time.time()

    out_dict = static_canon(p, q)

    t1 = time.time()
    log.info(f"finished static canonicalization - took {t1-t0} seconds")

    M = out_dict["M"]
    algo_factor = out_dict["algo_factor"]
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
    # solver = scs.SCS(data, cones_dict, eps_abs=tol_abs, eps_rel=tol_rel)
    solver = scs.SCS(data, cones_dict, eps_abs=tol_abs, eps_rel=tol_rel, 
                verbose=True,
                normalize=False,
                max_iters=int(1e5),
                scale=0.1,
                adaptive_scale=False,
                alpha=1.0,
                rho_x=1e-6,
                acceleration_lookback=0)
    solve_times = np.zeros(N)
    x_stars = jnp.zeros((N, n))
    y_stars = jnp.zeros((N, m))
    s_stars = jnp.zeros((N, m))
    q_mat = jnp.zeros((N, m + n))

    """
    sample theta and get y for each problem
    """

    batch_q = vmap(single_q, in_axes=(0, None, None, None, None), out_axes=(0))

    m, n = A_sparse.shape
    q_mat = batch_q(thetas, m, n, p, q)

    scs_instances = []
    for i in range(N):
        log.info(f"solving problem number {i}")

        # update
        b_np = np.array(q_mat[i, n:])
        c_np = np.array(q_mat[i, :n])

        # manual canon
        manual_canon_dict = {
            "P": P_sparse,
            "A": A_sparse,
            "b": b_np,
            "c": c_np,
            "cones": cones_dict,
        }
        scs_instance = SCSinstance(manual_canon_dict, solver, manual_canon=True)

        scs_instances.append(scs_instance)
        x_stars = x_stars.at[i, :].set(scs_instance.x_star)
        y_stars = y_stars.at[i, :].set(scs_instance.y_star)
        s_stars = s_stars.at[i, :].set(scs_instance.s_star)
        q_mat = q_mat.at[i, :].set(scs_instance.q)
        solve_times[i] = scs_instance.solve_time
        # pdb.set_trace()

        # check with our jax implementation
        # P_jax = jnp.array(P_sparse.todense())
        # A_jax = jnp.array(A_sparse.todense())
        # c_jax, b_jax = jnp.array(c), jnp.array(b)
        # data = dict(P=P_jax, A=A_jax, b=b_jax, c=c_jax, cones=cones_dict)
        # # data['x'] = x_stars[i, :]
        # # data['y'] = y_stars[i, :]
        # x_jax, y_jax, s_jax = scs_jax(data, iters=1000)

        ############
        # qq = single_q(x_init_mat[0, :], m, n, cfg.T, cfg.nx, cfg.nu, cfg.state_box, cfg.control_box, Ad)

    # resave the data??
    # print('saving final data...', flush=True)
    log.info("saving final data...")
    t0 = time.time()
    jnp.savez(
        output_filename,
        thetas=thetas,
        x_stars=x_stars,
        y_stars=y_stars,
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

    for i in range(5):
        plt.plot(y_stars[i, :])
    plt.savefig("y_stars.pdf")
    plt.clf()


    # save plot of first 5 parameters
    for i in range(5):
        plt.plot(thetas[i, :])
    plt.savefig("thetas.pdf")

    x_dist = distance_matrix(x_stars, x_stars)
    y_dist = distance_matrix(y_stars, y_stars)
    pdb.set_trace()


def static_canon(p, q):
    # nothing to do with theta
    # just need the appropriate sizes
    # don't need to do this in jax
    # create cvxpy problem
    # M has size (p, q)

    L = cp.Variable((p, q))
    S = cp.Variable((p, q))
    theta = sample_theta(p, q, .1, 2)
    mu = theta[0]
    M_vec = theta[1:]
    M = np.reshape(M_vec, (p, q))

    constraints = [L + S == M, cp.sum(cp.abs(S)) <= mu]
    obj = cp.Minimize(cp.norm(L, 'nuc'))
    prob = cp.Problem(obj, constraints)
    # sol = prob.solve(verbose=True)
    data, _, __ = prob.get_problem_data(cp.SCS)

    A, b, c = data['A'], data['b'], data['c']
    m, n = A.shape
    P = np.zeros((n, n))

    P_jax = jnp.array(P)
    A_jax = jnp.array(A.todense())
    M = jnp.zeros((n + m, n + m))
    M = M.at[:n, :n].set(P_jax)
    M = M.at[:n, n:].set(A_jax.T)
    M = M.at[n:, :n].set(-A_jax)

    # factor for DR splitting
    algo_factor = jsp.linalg.lu_factor(M + jnp.eye(n + m))

    A_sparse = csc_matrix(A)
    P_sparse = csc_matrix(P)
    cones_cp = data['dims']
    cones = {'z': cones_cp.zero, 'l': cones_cp.nonneg, 'q': cones_cp.soc, 's': cones_cp.psd}
    out_dict = dict(
        M=M,
        algo_factor=algo_factor,
        cones_dict=cones,
        # cones_array=cones_array,
        A_sparse=A_sparse,
        P_sparse=P_sparse,
        b=b,
        c=c
        # A_dynamics=Ad,
    )
    # data_scs = {
    #     'P': csc_matrix(np.zeros((n, n))),
    #     'A': data['A'],
    #     'b': data['b'],
    #     'c': data['c'],
    # }
    # soln = scs.solve(data_scs, cones, verbose=True)
    # pdb.set_trace()
    return out_dict


def low_2_high_dim_prediction(nn_output, X_list, Y_list, n_x_low, n_y_low,
                              n_x_non_psd, n_y_non_psd, dx, dy, x_psd_size, y_psd_size,
                              tx, ty, x_psd_indices):
    '''
    theta -> [NN] -> nn_output

    nn_output = concat(x_low_dim, y_low_dim, alpha)

    x_low_dim = concat(x_non_psd, u_1, ..., u_d)
    y_low_dim = concat(x_non_psd, v_1, ..., v_d)

    X_psd = sum_{i=1}^d u_i u_i^T + sum_{i=1}^t alpha_i X_list[i]
    Y_psd = sum_{i=1}^d v_i v_i^T + sum_{i=1}^t alpha_i Y_list[i]

    x = concat(x_non_psd, vec_symm(X_psd))
    y = concat(y_non_psd, vec_symm(Y_psd))

    return concat([x, y])
    '''
    # pdb.set_trace()
    x_low_dim = nn_output[:n_x_low]
    y_low_dim = nn_output[n_x_low:n_x_low + n_y_low]
    alpha_x = nn_output[n_x_low + n_y_low:n_x_low + n_y_low + tx]
    alpha_y = nn_output[n_x_low + n_y_low + tx:]
    print('alpha_x', alpha_x)
    print('alpha_y', alpha_y)

    x_non_psd = x_low_dim[:n_x_non_psd]
    x_psd = x_low_dim[n_x_non_psd:]
    U = jnp.reshape(x_psd, (dx, x_psd_size))
    y_non_psd = y_low_dim[:n_y_non_psd]
    y_psd = y_low_dim[n_y_non_psd:]
    # pdb.set_trace()
    V = jnp.reshape(y_psd, (dy, y_psd_size))

    sum_uuT = jnp.zeros((x_psd_size, x_psd_size))
    for i in range(dx):
        sum_uuT = sum_uuT + jnp.outer(U[i, :], U[i, :])
    sum_vvT = jnp.zeros((y_psd_size, y_psd_size))
    for i in range(dy):
        sum_vvT = sum_vvT + jnp.outer(V[i, :], V[i, :])
    sum_alpha_X = jnp.zeros((x_psd_size, x_psd_size))
    for i in range(tx):
        sum_alpha_X = sum_alpha_X + alpha_x[i] * X_list[i] / alpha_x.sum()
    sum_alpha_Y = jnp.zeros((y_psd_size, y_psd_size))
    for i in range(ty):
        sum_alpha_Y = sum_alpha_Y + alpha_y[i] * Y_list[i] / alpha_y.sum()
    # sum_uuT = jnp.sum([jnp.outer(U[i, :], U[i, :]) for i in range(dx)])
    # sum_vvT = jnp.sum([jnp.outer(V[i, :], V[i, :]) for i in range(dy)])
    # sum_alpha_X = jnp.sum([alpha_x * X_list[i] for i in range(tx)])
    # sum_alpha_Y = jnp.sum([alpha_y * Y_list[i] for i in range(ty)])
    print('sum_alpha_X', sum_alpha_X)
    print('sum_alpha_Y', sum_alpha_Y)
    X_psd = sum_uuT + sum_alpha_X  #+ 10 * jnp.eye(x_psd_size)
    Y_psd = sum_vvT + sum_alpha_Y  #+ 10 * jnp.eye(x_psd_size)
    X_vec = vec_symm(X_psd)
    Y_vec = vec_symm(Y_psd)
    print('X_vec', X_vec)
    print('Y_vec', Y_vec)

    # x = jnp.concatenate([X_vec, x_non_psd])
    n = X_vec.size + x_non_psd.size
    x = jnp.zeros(n)

    x = x.at[x_psd_indices].set(X_vec)
    x_non_psd_indices = jnp.arange(n)
    x_non_psd_indices = jnp.delete(x_non_psd_indices, x_psd_indices)
    x = x.at[x_non_psd_indices].set(x_non_psd)
    y = jnp.concatenate([y_non_psd, Y_vec])

    return jnp.concatenate([x, y])


def single_q(thetas, m, n, p, q):
    vec_M = thetas[1:]
    mu = thetas[0]
    M = jnp.reshape(vec_M, (p, q))

    b = jnp.zeros(m)
    b = b.at[:p*q].set(jnp.ravel(M.T))
    b = b.at[3*p*q].set(mu)

    c = jnp.zeros(n)
    # psd_size = p + q

    # vec_c_size = int(psd_size * (psd_size + 1) / 2)
    # eye = jnp.eye(psd_size)
    # I_vec = eye[jnp.triu_indices(psd_size)]
    vec_c1_size = int(p * (p + 1) / 2)
    eye1 = jnp.eye(p)
    I1_vec = eye1[jnp.triu_indices(p)]

    vec_c2_size = int(q * (q + 1) / 2)
    eye2 = jnp.eye(q)
    I2_vec = eye2[jnp.triu_indices(q)]

    c = c.at[:vec_c1_size].set(0.5 * I1_vec)
    c = c.at[vec_c1_size:vec_c1_size+vec_c2_size].set(0.5 * I2_vec)
    # m, nvars = 0, 0
    qvec = jnp.zeros(m + n)
    qvec = qvec.at[:n].set(c)
    qvec = qvec.at[n:].set(b)

    return qvec


def symmvec(A):
    # todo
    return jnp.triu(A)


if __name__ == "__main__":
    test_robust_pca()
