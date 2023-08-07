import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp
from l2ws.scs_problem import SCSinstance
import pdb
import cvxpy as cp
from scipy.sparse import csc_matrix, save_npz, load_npz
import osqp


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
    }
)
log = logging.getLogger(__name__)


def save_results_dynamic(output_filename, theta_mat, z_stars, q_mat, factors, ref_traj_tensor=None):
    """
    saves the results from the setup phase
    saves q_mat in csc_matrix form to save space
    everything else is saved as a npz file
    also plots z_stars, q, thetas
    """

    # save theta_mat, z_stars, factors
    #   needs to save factors[0] and factors[1] separately
    t0 = time.time()
    if ref_traj_tensor is None:
        jnp.savez(
            output_filename,
            thetas=jnp.array(theta_mat),
            z_stars=z_stars#,
            # factors0=factors[0],
            # factors1=factors[1]
        )
    else:
        jnp.savez(
            output_filename,
            thetas=jnp.array(theta_mat),
            z_stars=z_stars,
            ref_traj_tensor=ref_traj_tensor
            #,
            # factors0=factors[0],
            # factors1=factors[1],
            # ref_traj_tensor=ref_traj_tensor
        )
    # ref_traj_tensor has shape (num_rollouts, num_goals, goal_length)
    t1 = time.time()
    print('time to save non-sparse', t1 - t0)

    # save the q_mat but as a sparse object
    t2 = time.time()
    q_mat_sparse = csc_matrix(q_mat)
    save_npz(f"{output_filename}_q", q_mat_sparse)
    t3 = time.time()
    print('time to save non-sparse', t3 - t2)

    # save plot of first 5 solutions
    for i in range(5):
        plt.plot(z_stars[i, :])
    plt.savefig("z_stars.pdf")
    plt.clf()

    # save plot of first 5 q
    for i in range(5):
        plt.plot(q_mat[i, :])
    plt.savefig("q.pdf")
    plt.clf()

    # save plot of first 5 parameters
    for i in range(5):
        plt.plot(theta_mat[i, :])
    plt.savefig("thetas.pdf")
    plt.clf()


def load_results_dynamic(output_filename):
    """
    returns the saved results from the corresponding save_results_dynamic function
    """
    q_mat_sparse = load_npz(f"{output_filename}_q")
    loaded_obj = jnp.load(output_filename)
    theta_mat, z_stars = loaded_obj['thetas'], loaded_obj['z_stars']
    factors0, factors1 = loaded_obj['factors0'], loaded_obj['factors1']
    factors = (factors0, factors1)
    return theta_mat, z_stars, q_mat_sparse, factors



def direct_osqp_setup_script(theta_mat, q_mat, P, A, output_filename, z_stars=None):
    # def solve_many_probs_cvxpy(A, b_mat, lambd):
    """
    solves many lasso problems where each problem has a different b vector
    """
    # import pdb
    # pdb.set_trace()
    m, n = A.shape
    N = q_mat.shape[0]

    # P, A
    osqp_solver = osqp.OSQP()
    P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))
    c, l, u = np.zeros(n), np.zeros(m), np.zeros(m)
    osqp_solver.setup(P=P_sparse, q=c, A=A_sparse, l=l, u=u,
                        max_iter=2000, verbose=True, eps_abs=1e-5, eps_rel=1e-5)

    solve_times = np.zeros(N)
    if z_stars is None:
        z_stars = jnp.zeros((N, n + 2 * m))
        objvals = jnp.zeros((N))
        x_stars = []
        y_stars = []
        w_stars = []
        for i in range(N):
            log.info(f"solving problem number {i}")

            # setup c, l, u
            c, l, u = q_mat[i, :n], q_mat[i, n:n + m], q_mat[i, n + m:]
            osqp_solver.update(q=np.array(c))
            osqp_solver.update(l=np.array(l), u=np.array(u))

            # solve with osqp
            results = osqp_solver.solve()

            # set the solve time in seconds
            solve_times[i] = results.info.solve_time
            # solve_iters[i] = results.info.iter

            # set the results
            # x_sols = x_sols.at[i, :].set(results.x)
            # y_sols = y_sols.at[i, :].set(results.y)

            x_stars.append(results.x)
            y_stars.append(results.y)
            w_stars.append(A @ results.x)
            # z_stars = z_stars.at[i, :n].set(results.x)
            # z_stars = z_stars.at[i, n:n + m].set(results.y)
            # z_stars = z_stars.at[i, n + m:].set(A @ results.x)
            # import pdb
            # pdb.set_trace()
            
            
            # objvals = objvals.at[i].set(prob.value)

            # x_star = jnp.array(x.value)
            # y_star = jnp.array(constraints[0].dual_value)
            # z_star = jnp.concatenate([x_star, y_star])
            # z_stars = z_stars.at[i, :].set(z_star)
            # solve_times[i] = prob.solver_stats.solve_time
    z_stars = jnp.hstack([jnp.stack(x_stars), jnp.stack(y_stars), jnp.stack(w_stars)])

            if i % 1000 == 0:
                # save the data
                log.info("saving data...")
                t0 = time.time()
                jnp.savez(
                    output_filename,
                    thetas=jnp.array(theta_mat),
                    z_stars=z_stars,
                    q_mat=q_mat
                )

    # save the data
    log.info("final saving final data...")
    t0 = time.time()
    jnp.savez(
        output_filename,
        thetas=jnp.array(theta_mat),
        z_stars=z_stars,
        q_mat=q_mat
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

    # save plot of first 5 q
    for i in range(5):
        plt.plot(q_mat[i, :])
    plt.savefig("q.pdf")
    plt.clf()


    # save plot of first 5 parameters
    for i in range(5):
        plt.plot(theta_mat[i, :])
    plt.savefig("thetas.pdf")
    plt.clf()

    return z_stars


def osqp_setup_script(theta_mat, q_mat, P, A, output_filename, z_stars=None):
    # def solve_many_probs_cvxpy(A, b_mat, lambd):
    """
    solves many lasso problems where each problem has a different b vector
    """
    m, n = A.shape
    N = q_mat.shape[0]

    # setup cvxpy
    x, w = cp.Variable(n), cp.Variable(m)
    c_param, l_param, u_param = cp.Parameter(n), cp.Parameter(m), cp.Parameter(m)
    constraints = [A @ x == w, l_param <= w, w <= u_param]
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, P) + c_param @ x), constraints)

    solve_times = np.zeros(N)
    if z_stars is None:
        z_stars = jnp.zeros((N, n + m))
        objvals = jnp.zeros((N))
        for i in range(N):
            log.info(f"solving problem number {i}")

            # solve with cvxpy
            c_param.value = np.array(q_mat[i, :n])
            l_param.value = np.array(q_mat[i, n:n + m])
            u_param.value = np.array(q_mat[i, n + m:])
            prob.solve(verbose=True, solver=cp.OSQP, eps_abs=1e-03, eps_rel=1e-03)
            objvals = objvals.at[i].set(prob.value)

            x_star = jnp.array(x.value)
            y_star = jnp.array(constraints[0].dual_value)
            z_star = jnp.concatenate([x_star, y_star])
            z_stars = z_stars.at[i, :].set(z_star)
            solve_times[i] = prob.solver_stats.solve_time

    # save the data
    log.info("final saving final data...")
    t0 = time.time()
    jnp.savez(
        output_filename,
        thetas=jnp.array(theta_mat),
        z_stars=z_stars,
        q_mat=q_mat
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

    # save plot of first 5 q
    for i in range(5):
        plt.plot(q_mat[i, :])
    plt.savefig("q.pdf")
    plt.clf()


    # save plot of first 5 parameters
    for i in range(5):
        plt.plot(theta_mat[i, :])
    plt.savefig("thetas.pdf")
    plt.clf()


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
        print('solving problem', i)
        b_param.value = np.array(b_mat[i, :])
        prob.solve(verbose=False)
        objvals = objvals.at[i].set(prob.value)
        z_stars = z_stars.at[i, :].set(jnp.array(z.value))
        solve_times[i] = prob.solver_stats.solve_time

    # save the data
    log.info("final saving final data...")
    t0 = time.time()
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

    return x_stars, y_stars, s_stars
