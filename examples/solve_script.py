import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp
from l2ws.scs_problem import SCSinstance
import pdb
import cvxpy as cp


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
    }
)
log = logging.getLogger(__name__)


def ista_setup_script(b_mat, A, lambd, output_filename):
    # def solve_many_probs_cvxpy(A, b_mat, lambd):
    """
    solves many lasso problems where each problem has a different b vector
    """
    m, n = A.shape
    N = b_mat.shape[0]
    z, b_param = cp.Variable(n), cp.Parameter(m)
    prob = cp.Problem(cp.Minimize(.5 * cp.sum_squares(np.array(A) @ z - b_param) + lambd * cp.norm(z, p=1)))
    z_stars = jnp.zeros((N, n))
    objvals = jnp.zeros((N))
    solve_times = np.zeros(N)
    for i in range(N):
        b_param.value = np.array(b_mat[i, :])
        prob.solve(verbose=False)
        objvals = objvals.at[i].set(prob.value)
        z_stars = z_stars.at[i, :].set(jnp.array(z.value))
        solve_times[i] = prob.solver_stats.solve_time

    # save the data
    log.info("final saving final data...")
    t0 = time.time()
    # import pdb
    # pdb.set_trace()
    jnp.savez(
        output_filename,
        thetas=jnp.array(b_mat),
        z_stars=z_stars,
    )

    # save solve times
    df_solve_times = pd.DataFrame(solve_times, columns=['solve_times'])
    df_solve_times.to_csv('solve_times.csv')

    save_time = time.time()
    log.info(f"finished saving final data... took {save_time-t0}'")

    # save plot of first 5 solutions
    for i in range(5):
        plt.plot(z_stars[i, :])
    plt.savefig("z_stars.pdf")
    plt.clf()


    # save plot of first 5 parameters
    for i in range(5):
        plt.plot(b_mat[i, :])
    plt.savefig("thetas.pdf")
    plt.clf()



def setup_script(q_mat, theta_mat, solver, data, cones_dict, output_filename, solve=True):
    N = q_mat.shape[0]
    m, n = data['A'].shape

    solve_times = np.zeros(N)
    x_stars = jnp.zeros((N, n))
    y_stars = jnp.zeros((N, m))
    s_stars = jnp.zeros((N, m))
    # q_mat = jnp.zeros((N, m + n))
    # scs_instances = []

    P_sparse, A_sparse = data['P'], data['A']
    if solve:
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

            # scs_instances.append(scs_instance)
            x_stars = x_stars.at[i, :].set(scs_instance.x_star)
            y_stars = y_stars.at[i, :].set(scs_instance.y_star)
            s_stars = s_stars.at[i, :].set(scs_instance.s_star)
            q_mat = q_mat.at[i, :].set(scs_instance.q)
            solve_times[i] = scs_instance.solve_time

            if i % 1000 == 0:
                log.info(f"saving final data... after solving problem number {i}")
                jnp.savez(
                    output_filename,
                    thetas=theta_mat,
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
        thetas=theta_mat,
        x_stars=x_stars,
        y_stars=y_stars,
        s_stars=s_stars,
        q_mat=q_mat
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
        plt.plot(theta_mat[i, :])
    plt.savefig("thetas.pdf")
    plt.clf()
