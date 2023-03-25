import numpy as np
import logging
import yaml
from jax import vmap
import pandas as pd
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp
import os
import scs
import cvxpy as cp
import jax.scipy as jsp
from l2ws.algo_steps import create_M


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
    }
)
log = logging.getLogger(__name__)


def generate_A_tensor(N, n_orig, r):
    """
    generates covariance matrices A_1, ..., A_N
        where each A_i has shape (n_orig, n_orig)
    A_i = F Sigma_i F^T
        where F has shape (n_orig, r)
    i.e. each Sigma_i is psd (Sigma_i = B_i B_i^T) and is different
        B_i has shape (r, r)
        F stays the same for each problem
    We let theta = upper_tri(Sigma_i)
    """
    F = np.random.rand(n_orig, r)
    A_tensor = np.zeros((N, n_orig, n_orig))
    n_choose_2 = int(n_orig * (n_orig + 1) / 2)
    theta_mat = np.zeros((N, n_choose_2))
    for i in range(N):
        B = np.random.rand(r, r)
        Sigma = B @ B.T
        theta_mat[i, :] = np.triu(Sigma)
        A_tensor[i, :, :] = F @ Sigma @ F.T
    return A_tensor, theta_mat


def cvxpy_prob(n_orig, k):
    A_param = cp.Parameter((n_orig, n_orig), symmetric=True)
    X = cp.Variable((n_orig, n_orig), symmetric=True)
    constraints = [cp.sum(cp.abs(X)) <= k, cp.trace(X) == 1]
    prob = cp.Problem(cp.trace(A_param @ X), constraints)
    return prob, A_param


def get_q_mat(A_tensor, prob, A_param, m, n):
    N, n_orig, _ = A_tensor.shape
    q_mat = jnp.zeros((N, m + n))
    for i in range(N):
        # set the parameter
        A_param.value = A_tensor[i, :, :]

        # get the problem data
        data, _, __ = prob.get_problem_data(cp.SCS)

        c, b = data['c'], data['b']
        n = c.size
        q_mat = q_mat.at[i, :n].set(c)
        q_mat = q_mat.at[i, n:].set(b)
    return q_mat


def static_canon(n_orig, k):
    # create the cvxpy problem
    prob, A_param = cvxpy_prob(n_orig, k)

    # get the problem data
    data, _, __ = prob.get_problem_data(cp.SCS)

    A_sparse, c, b = data['A'], data['c'], data['b']
    P_sparse = jnp.zeros
    cones_cp = data['dims']

    # factor for DR splitting
    m, n = A_sparse.shape
    M = create_M(P_sparse, A_sparse)
    algo_factor = jsp.linalg.lu_factor(M + jnp.eye(n + m))

    # set the dict
    cones = {'z': cones_cp.zero, 'l': cones_cp.nonneg, 'q': cones_cp.soc, 's': cones_cp.psd}
    out_dict = dict(
        M=M,
        algo_factor=algo_factor,
        cones_dict=cones,
        A_sparse=A_sparse,
        P_sparse=P_sparse,
        b=b,
        c=c,
        prob=prob,
        A_param=A_param
    )
    return out_dict


def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    m_orig, n_orig = cfg.m_orig, cfg.n_orig

    # np.random.seed(cfg.seed)

    log.info("creating static canonicalization...")
    t0 = time.time()
    out_dict = static_canon(cfg.n_orig, cfg.k)

    t1 = time.time()
    log.info(f"finished static canonicalization - took {t1-t0} seconds")

    cones_dict = out_dict["cones_dict"]
    A_sparse, P_sparse = out_dict["A_sparse"], out_dict["P_sparse"]
    b, c = out_dict["b"], out_dict["c"]
    prob, A_param = out_dict["prob"], out_dict["A_param"]
    m, n = A_sparse.shape

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    # create scs solver object
    #    we can cache the factorization if we do it like this
    data = dict(P=P_sparse, A=A_sparse, b=b, c=c)
    tol_abs = cfg.solve_acc_abs
    tol_rel = cfg.solve_acc_rel
    solver = scs.SCS(data, cones_dict, eps_abs=tol_abs, eps_rel=tol_rel)
    solve_times = np.zeros(N)
    x_stars = jnp.zeros((N, n))
    y_stars = jnp.zeros((N, m))
    s_stars = jnp.zeros((N, m))
    q_mat = jnp.zeros((N, m + n))

    # sample theta for each problem -- generate A_tensor from factor model
    thetas_np = (2 * np.random.rand(N, m_orig) - 1) * cfg.b_range + cfg.b_nominal
    thetas = jnp.array(thetas_np)

    q_mat = get_q_mat(A_tensor, prob, A_param, m, n)

    scs_instances = []

    if setup_cfg['solve']:
        for i in range(N):
            log.info(f"solving problem number {i}")

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

            if i % 1000 == 0:
                log.info(f"saving final data... after solving problem number {i}")
                jnp.savez(
                    output_filename,
                    thetas=thetas,
                    x_stars=x_stars,
                    y_stars=y_stars,
                    s_stars=s_stars,
                    q_mat=q_mat
                )
        # save the data
        log.info("final saving final data...")
        t0 = time.time()
        jnp.savez(
            output_filename,
            thetas=thetas,
            x_stars=x_stars,
            y_stars=y_stars,
            s_stars=s_stars,
            q_mat=q_mat
        )
    else:
        log.info("final saving final data...")
        t0 = time.time()
        jnp.savez(
            output_filename,
            thetas=thetas,
            q_mat=q_mat,
            m=m,
            n=n
        )

    # save solve times
    df_solve_times = pd.DataFrame(solve_times, columns=['solve_times'])
    df_solve_times.to_csv('solve_times.csv')

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
