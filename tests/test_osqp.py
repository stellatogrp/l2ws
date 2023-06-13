import time
import jax.numpy as jnp
import scs
import numpy as np
from scipy.sparse import csc_matrix
from l2ws.algo_steps import k_steps_eval_osqp, k_steps_train_osqp, create_projection_fn, lin_sys_solve, k_steps_train_osqp, k_steps_eval_scs
import jax.scipy as jsp
import pytest
import matplotlib.pyplot as plt
from l2ws.osqp_model import OSQPmodel
import cvxpy as cp
from jax import vmap
from functools import partial
from examples.osc_mass import multiple_random_osc_mass_osqp
from examples.mpc import multiple_random_mpc_osqp, solve_many_probs_cvxpy, solve_multiple_trajectories, shifted_sol
from examples.mnist import get_mnist, vectorized2DBlurMatrix, mnist_canon
from scipy.spatial import distance_matrix
# from examples.ista import sol_2_obj_diff, solve_many_probs_cvxpy
from l2ws.utils.nn_utils import get_nearest_neighbors
import osqp


# def solve_many_osqp(P, A, q_mat):
#     m, n = A.shape
#     N, p = q_mat.shape
#     for i in range(N):

@pytest.mark.skip(reason="temp")
def test_mnist():
    # load mnist data
    x_train, x_test = get_mnist()

    # create A matrix filter
    A = vectorized2DBlurMatrix(28, 28, 8)

    # blur img
    blurred_img = np.reshape(A @ x_train[0, :], (28, 28))

    # create cvxpy problem with TV regularization

    # get P, A, q, l, u with cvxpy osqp canonicalization
    lambd = 1e-4
    P, A, c, l, u = mnist_canon(A, lambd, blurred_img)
    q = jnp.concatenate([c, l, u])

    # solve with our osqp
    m, n = A.shape
    k = 2000
    sigma = 1e-6
    rho_vec = jnp.ones(m)

    rho_vec = rho_vec.at[l == u].set(1000)

    # M = P + sigma * jnp.eye(n) + rho * A.T @ A
    M = P + sigma * jnp.eye(n) + A.T @ jnp.diag(rho_vec) @ A

    factor = jsp.linalg.lu_factor(M)

    z_final, iter_losses, z_all_plus_1, primal_resids, dual_resids = k_steps_eval_osqp(k, np.zeros(
        m + n), q, factor, P, A, rho_vec, sigma=sigma, supervised=False, z_star=None, jit=True)

    img = z_final[:784]

    plt.imshow(np.reshape(img, (28, 28)))
    plt.show()

    plt.plot(iter_losses)
    plt.yscale('log')
    plt.show()
    import pdb
    pdb.set_trace()


# @pytest.mark.skip(reason="temp")
def test_quadcopter():
    N_train = 200
    N_test = 100
    N = N_train + N_test
    T = 10
    
    traj_length = 20
    num_traj = int(N_test / traj_length)
    num_traj_train = int(N_train / traj_length)
    x_init_factor = 10
    noise_std_dev = .00 #0.05
    nx, nu = 12, 4

    mpc_setup = multiple_random_mpc_osqp(N_train,
                                         T=T,
                                         nx=nx,
                                         nu=nu,
                                         Ad=None,
                                         Bd=None,
                                         seed=42,
                                         x_init_factor=x_init_factor,
                                         quadcopter=True)
    factor, P, A, q_mat_train, theta_mat_train, x_min, x_max, Ad, Bd, rho_vec = mpc_setup
    m, n = A.shape
    q = q_mat_train[0, :]

    theta_mat_train, z_stars_train, q_mat_train = solve_multiple_trajectories(
        traj_length, num_traj_train, x_min, x_max, x_init_factor, Ad, P, A, q, noise_std_dev)

    theta_mat_test, z_stars_test, q_mat_test = solve_multiple_trajectories(
        traj_length, num_traj, x_min, x_max, x_init_factor, Ad, P, A, q, noise_std_dev)

    # create theta_mat and q_mat
    q_mat = jnp.vstack([q_mat_train, q_mat_test])
    theta_mat = jnp.vstack([theta_mat_train, theta_mat_test])

    # solve the QPs
    # z_stars, objvals = solve_many_probs_cvxpy(P, A, q_mat)
    # z_stars_train, z_stars_test = z_stars[:N_train, :], z_stars[N_train:, :]

    train_unrolls = 15
    input_dict = dict(rho=rho_vec,
                      q_mat_train=q_mat_train,
                      q_mat_test=q_mat_test,
                      P=P,
                      A=A,
                      factor=factor,
                      train_inputs=theta_mat[:N_train, :],
                      test_inputs=theta_mat[N_train:, :],
                      train_unrolls=train_unrolls,
                      nn_cfg={'intermediate_layer_sizes': [1000]},
                      supervised=False,
                      z_stars_train=z_stars_train,
                      z_stars_test=z_stars_test,
                      jit=True)
    osqp_model = OSQPmodel(input_dict)

    sim_len = 20
    q_init = q_mat_train[0, :] #q_mat_test[0, :]
    x_init = theta_mat_train[0, :] #theta_mat_test[0, :]
    k_plot = train_unrolls
    noise_vec_list = [noise_std_dev * jnp.array(np.random.normal(size=x_init.size)) for i in range(sim_len)]

    # simulate forward
    opt_sols_cs, state_traj_cs = simulate_fwd_l2ws(sim_len, osqp_model, k_plot, noise_vec_list, q_init, x_init, A, Ad, Bd, T, nx, nu)

    # z_prev = shifted_sol(opt_sols_learned[0][:m + n], T, nx, nu, m, n)
    opt_sols_opt, state_traj_opt = simulate_fwd_l2ws(sim_len, osqp_model, 275, noise_vec_list, q_init, x_init, A, Ad, Bd, T, nx, nu)
    opt_sols_prev_sol, state_traj_prev = simulate_fwd_l2ws(sim_len, osqp_model, k_plot, noise_vec_list, q_init, x_init, A, Ad, Bd, T, nx, nu, prev_sol=True)

    # do the plotting
    state_traj_list = [state_traj_opt, state_traj_cs, state_traj_prev]
    labels = ['optimal', 'cold-start', 'prev_sol']
    plot_traj_3d(state_traj_list, labels)
    


    train_inputs, test_inputs = theta_mat_train, theta_mat_test

    # full evaluation on the test set with nearest neighbor
    k = 100
    nearest_neighbors_z = get_nearest_neighbors(train_inputs, test_inputs, z_stars_train[:,:m+n])
    nn_eval_out = osqp_model.evaluate(k, nearest_neighbors_z,
                                      q_mat_test, z_stars=None,
                                      fixed_ws=True, tag='test')
    nn_losses = nn_eval_out[1][1].mean(axis=0)

    # full evaluation on the test set with prev solution
    non_first_indices = jnp.mod(jnp.arange(N_test), num_traj) != 0
    non_last_indices = jnp.mod(jnp.arange(N_test), num_traj) != num_traj - 1
    q_mat_prev = q_mat_test[non_first_indices, :]

    # batch_shifted_sol = vmap(shifted_sol_partial, in_axes=(0,), out_axes=(0,))
    partial_shifted_sol_fn = partial(shifted_sol, T=T, nx=nx, nu=nu, m=m, n=n)
    batch_shifted_sol_fn = vmap(partial_shifted_sol_fn, in_axes=(0), out_axes=(0))

    prev_z = batch_shifted_sol_fn(z_stars_test[non_last_indices, :m+n])
    prev_sol_out = osqp_model.evaluate(k, prev_z,
                                       q_mat_prev, z_stars=None,
                                       fixed_ws=True, tag='test')
    prev_sol_losses = prev_sol_out[1][1].mean(axis=0)

    # full evaluation on the test set with cold-start
    init_eval_out = osqp_model.evaluate(
        k, test_inputs, q_mat_test, z_stars=z_stars_test, fixed_ws=False, tag='test')
    init_test_losses = init_eval_out[1][1].mean(axis=0)
        # train the osqp_model
    # call train_batch without jitting
    params, state = osqp_model.params, osqp_model.state
    num_epochs = 1000
    train_losses = jnp.zeros(num_epochs)
    # import pdb
    # pdb.set_trace()
    for i in range(num_epochs):
        train_result = osqp_model.train_full_batch(params, state)
        loss, params, state = train_result
        train_losses = train_losses.at[i].set(loss)
        print('loss', i, loss)

    osqp_model.params, osqp_model.state = params, state

    # full evaluation on the test set
    final_eval_out = osqp_model.evaluate(
        k, test_inputs, q_mat_test, z_stars=z_stars_test, fixed_ws=False, tag='test')
    final_test_losses = final_eval_out[1][1].mean(axis=0)

    # plotting
    plt.plot(init_test_losses, label='cold start')
    plt.plot(final_test_losses, label=f"learned warm-start k={train_unrolls}")
    plt.plot(prev_sol_losses, label='prev sol')
    plt.plot(nn_losses, label='nearest neighbor')
    plt.yscale('log')
    plt.legend()
    plt.show()

    # simulate forward
    opt_sols_learned, state_traj_learned = simulate_fwd_l2ws(sim_len, osqp_model, k_plot, noise_vec_list, q_init, x_init, A, Ad, Bd, T, nx, nu)
    opt_sols_opt, state_traj_opt = simulate_fwd_l2ws(sim_len, osqp_model, 275, noise_vec_list, q_init, x_init, A, Ad, Bd, T, nx, nu)

    # # do the plotting
    # state_traj_list = [state_traj_opt, state_traj_learned]
    # labels = ['optimal', 'learned']
    # plot_traj_3d(state_traj_list, labels)

    # do the plotting
    state_traj_list = [state_traj_opt, state_traj_learned, state_traj_prev]
    labels = ['optimal', 'learned', 'prev_sol']
    plot_traj_3d(state_traj_list, labels)

    import pdb
    pdb.set_trace()

def plot_traj_3d(state_traj_list, labels):
    """
    state_traj_list is a list of lists
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for j in range(len(labels)):
        curr_state_traj = state_traj_list[j]
        x = np.array([curr_state_traj[i][0] for i in range(len(curr_state_traj))])
        y = np.array([curr_state_traj[i][1] for i in range(len(curr_state_traj))])
        z = np.array([curr_state_traj[i][2] for i in range(len(curr_state_traj))])
        if labels[j] == 'optimal':
            ax.scatter(x, y, z, label=labels[j], color='green')
        else:
            ax.plot(x, y, z, label=labels[j])
    plt.legend()
    plt.show()


def simulate_fwd_l2ws(sim_len, l2ws_model, k, noise_vec_list, q_init, x_init, A, Ad, Bd, T, nx, nu, prev_sol=False):
    """
    does the forward simulation

    returns
    """
    m, n = A.shape
    # get the first test_input and q_mat_test
    input = x_init
    q_mat = q_init

    opt_sols = []
    state_traj = [x_init]

    opt_sol = np.zeros(n + 2 * m)
    

    for i in range(sim_len):
        # evaluate
        if prev_sol:
            # get the shifted previous solution
            prev_z_shift = shifted_sol(opt_sol[:m + n], T, nx, nu, m, n)
            final_eval_out = l2ws_model.evaluate(
                k, prev_z_shift[None, :], q_mat[None, :], z_stars=None, fixed_ws=True, tag='test')
            # z_star = final_eval_out[1][2][0, -1, :]
        else:
            final_eval_out = l2ws_model.evaluate(
                k, input[None, :], q_mat[None, :], z_stars=None, fixed_ws=False, tag='test')
        print('loss', k, prev_sol, final_eval_out[1][0])

        # get the first control input
        # final_eval_out[1][2] will have shape (1, k, n + 2 * m)
        opt_sol = final_eval_out[1][2][0, -1, :]

        u0 = opt_sol[T * nx: T * nx + nu]

        # import pdb
        # pdb.set_trace()

        # input the first control to get the next state and perturb it
        x_init = Ad @ x_init + Bd @ u0 + noise_vec_list[i]

        # set test_input and q_mat_test
        input = x_init
        c, l, u = q_mat[:n], q_mat[n:n + m], q_mat[n + m:]
        Ad_x_init = Ad @ x_init
        l = l.at[:nx].set(-Ad_x_init)
        u = u.at[:nx].set(-Ad_x_init)
        q_mat = q_mat.at[n:n + m].set(l)
        q_mat = q_mat.at[n + m:].set(u)

        # append to the optimal solutions
        opt_sols.append(opt_sol)

        # append to the state trajectory
        state_traj.append(x_init)

    return opt_sols, state_traj


@pytest.mark.skip(reason="temp")
def test_shifted_sol():
    N_train = 10
    N_test = 10
    N = N_train + N_test
    T = 10
    num_traj = 10
    traj_length = 10
    x_init_factor = .5

    nx = 40
    nu = 20
    mpc_setup = multiple_random_mpc_osqp(N_train,
                                         T=T,
                                         nx=nx,
                                         nu=nu,
                                         Ad=None,
                                         Bd=None,
                                         seed=42,
                                         x_init_factor=x_init_factor)
    factor, P, A, q_mat_train, theta_mat_train, x_min, x_max, Ad, Bd, rho_vec = mpc_setup
    # train_inputs, test_inputs = theta_mat[:N_train, :], theta_mat[N_train:, :]
    # z_stars_train, z_stars_test = None, None
    # q_mat_train, q_mat_test = q_mat[:N_train, :], q_mat[N_train:, :]
    q = q_mat_train[0, :]

    # theta_mat_train, z_stars_train, q_mat_train = solve_multiple_trajectories(
    #     T, num_traj_train, x_bar, x_init_factor, Ad, P, A, q)
    noise_std_dev = 0
    theta_mat_test, z_stars_test, q_mat_test = solve_multiple_trajectories(
        traj_length, num_traj, x_min, x_max, x_init_factor, Ad, P, A, q, noise_std_dev)

    m, n = A.shape

    # get the shifted solution
    shifted_z_star = shifted_sol(z_stars_test[0, :], T, nx, nu, m, n)

    # warm-start with it
    k = 1000
    prev_sol_z_k, prev_sol_losses, prev_sol_z_all, prev_sol_pr, prev_sol_dr = k_steps_eval_osqp(k, shifted_z_star, q_mat_test[1, :], factor, P, A, rho_vec, sigma=1,
                                                                                                supervised=False, z_star=None, jit=True)
    z_k, losses, z_all, primal_resids, dual_resids = k_steps_eval_osqp(k, shifted_z_star * 0, q_mat_test[1, :], factor, P, A, rho_vec, sigma=1,
                                                                       supervised=False, z_star=None, jit=True)
    plt.title('state perturbation')
    plt.plot(losses, label='cold-start')
    plt.plot(prev_sol_losses, label='shifted prev sol warm-start')
    plt.yscale('log')
    plt.legend()
    plt.show()
    import pdb
    pdb.set_trace()

    assert dual_resids[-1] + primal_resids[-1] <= 1e-8
    assert dual_resids[0] + primal_resids[0] >= 1

    assert losses[200] >= prev_sol_losses[200] * 100

    assert jnp.linalg.norm(z_stars_test[0, nx:2*nx] - z_stars_test[1, :nx]) <= 1e-3
    assert jnp.linalg.norm(shifted_z_star[:nx] - z_stars_test[1, :nx]) <= 1e-3
    


@pytest.mark.skip(reason="temp")
def test_shift_train():
    N_train = 500
    N_test = 100
    N = N_train + N_test
    T = 10
    num_traj = 10
    traj_length = 10
    x_init_factor = .5
    noise_std_dev = 0

    mpc_setup = multiple_random_mpc_osqp(N_train,
                                         T=T,
                                         nx=10,
                                         nu=5,
                                         Ad=None,
                                         Bd=None,
                                         seed=42,
                                         x_init_factor=x_init_factor)
    factor, P, A, q_mat_train, theta_mat_train, x_min, x_max, Ad, Bd, rho_vec = mpc_setup
    # train_inputs, test_inputs = theta_mat[:N_train, :], theta_mat[N_train:, :]
    # z_stars_train, z_stars_test = None, None
    # q_mat_train, q_mat_test = q_mat[:N_train, :], q_mat[N_train:, :]
    q = q_mat_train[0, :]

    # theta_mat_train, z_stars_train, q_mat_train = solve_multiple_trajectories(
    #     T, num_traj_train, x_bar, x_init_factor, Ad, P, A, q)

    theta_mat_test, z_stars_test, q_mat_test = solve_multiple_trajectories(
        traj_length, num_traj, x_bar, x_init_factor, Ad, P, A, q, noise_std_dev)

    # create theta_mat and q_mat
    q_mat = jnp.vstack([q_mat_train, q_mat_test])
    theta_mat = jnp.vstack([theta_mat_train, theta_mat_test])

    # solve the QPs
    z_stars, objvals = solve_many_probs_cvxpy(P, A, q_mat)
    z_stars_train, z_stars_test = z_stars[:N_train, :], z_stars[N_train:, :]

    train_unrolls = 10
    input_dict = dict(rho=rho_vec,
                      q_mat_train=q_mat_train,
                      q_mat_test=q_mat_test,
                      P=P,
                      A=A,
                      factor=factor,
                      train_inputs=theta_mat[:N_train, :],
                      test_inputs=theta_mat[N_train:, :],
                      train_unrolls=train_unrolls,
                      nn_cfg={'intermediate_layer_sizes': [300]},
                      jit=True)
    osqp_model = OSQPmodel(input_dict)

    train_inputs, test_inputs = theta_mat_train, theta_mat_test

    # full evaluation on the test set with nearest neighbor
    k = 500
    nearest_neighbors_z = get_nearest_neighbors(train_inputs, test_inputs, z_stars_train)
    nn_eval_out = osqp_model.evaluate(k, nearest_neighbors_z,
                                      q_mat_test, z_stars=z_stars_test,
                                      fixed_ws=True, tag='test')
    nn_losses = nn_eval_out[1][1].mean(axis=0)


@pytest.mark.skip(reason="temp")
def test_mpc_prev_sol():
    N_train = 1000
    N_test = 100
    N = N_train + N_test
    T = 10
    num_traj = 10
    traj_length = 10
    x_init_factor = .5
    noise_std_dev = 0
    nx, nu = 10, 5

    mpc_setup = multiple_random_mpc_osqp(N_train,
                                         T=T,
                                         nx=nx,
                                         nu=nu,
                                         Ad=None,
                                         Bd=None,
                                         seed=42,
                                         x_init_factor=x_init_factor)
    factor, P, A, q_mat_train, theta_mat_train, x_min, x_max, Ad, Bd, rho_vec = mpc_setup
    m, n = A.shape
    q = q_mat_train[0, :]

    theta_mat_test, z_stars_test, q_mat_test = solve_multiple_trajectories(
        traj_length, num_traj, x_min, x_max, x_init_factor, Ad, P, A, q, noise_std_dev)

    # create theta_mat and q_mat
    q_mat = jnp.vstack([q_mat_train, q_mat_test])
    theta_mat = jnp.vstack([theta_mat_train, theta_mat_test])

    # solve the QPs
    z_stars, objvals = solve_many_probs_cvxpy(P, A, q_mat)
    z_stars_train, z_stars_test = z_stars[:N_train, :], z_stars[N_train:, :]

    train_unrolls = 10
    input_dict = dict(rho=rho_vec,
                      q_mat_train=q_mat_train,
                      q_mat_test=q_mat_test,
                      P=P,
                      A=A,
                      factor=factor,
                      train_inputs=theta_mat[:N_train, :],
                      test_inputs=theta_mat[N_train:, :],
                      train_unrolls=train_unrolls,
                      nn_cfg={'intermediate_layer_sizes': [300]},
                      jit=True)
    osqp_model = OSQPmodel(input_dict)

    train_inputs, test_inputs = theta_mat_train, theta_mat_test

    # full evaluation on the test set with nearest neighbor
    k = 500
    nearest_neighbors_z = get_nearest_neighbors(train_inputs, test_inputs, z_stars_train)
    nn_eval_out = osqp_model.evaluate(k, nearest_neighbors_z,
                                      q_mat_test, z_stars=z_stars_test,
                                      fixed_ws=True, tag='test')
    nn_losses = nn_eval_out[1][1].mean(axis=0)

    # full evaluation on the test set with prev solution
    non_first_indices = jnp.mod(jnp.arange(N_test), num_traj) != 0
    non_last_indices = jnp.mod(jnp.arange(N_test), num_traj) != num_traj - 1
    q_mat_prev = q_mat_test[non_first_indices, :]

    # batch_shifted_sol = vmap(shifted_sol_partial, in_axes=(0,), out_axes=(0,))
    partial_shifted_sol_fn = partial(shifted_sol, T=T, nx=nx, nu=nu, m=m, n=n)
    batch_shifted_sol_fn = vmap(partial_shifted_sol_fn, in_axes=(0), out_axes=(0))

    prev_z = batch_shifted_sol_fn(z_stars_test[non_last_indices, :])
    prev_sol_out = osqp_model.evaluate(k, prev_z,
                                       q_mat_prev, z_stars=None,
                                       fixed_ws=True, tag='test')
    prev_sol_losses = prev_sol_out[1][1].mean(axis=0)

    # full evaluation on the test set with cold-start
    init_eval_out = osqp_model.evaluate(
        k, test_inputs, q_mat_test, z_stars=z_stars_test, fixed_ws=False, tag='test')
    init_test_losses = init_eval_out[1][1].mean(axis=0)

    # train the osqp_model
    # call train_batch without jitting
    params, state = osqp_model.params, osqp_model.state
    num_epochs = 2
    train_losses = jnp.zeros(num_epochs)
    for i in range(num_epochs):
        train_result = osqp_model.train_full_batch(params, state)
        loss, params, state = train_result
        train_losses = train_losses.at[i].set(loss)

    osqp_model.params, osqp_model.state = params, state

    # full evaluation on the test set
    final_eval_out = osqp_model.evaluate(
        k, test_inputs, q_mat_test, z_stars=z_stars_test, fixed_ws=False, tag='test')
    final_test_losses = final_eval_out[1][1].mean(axis=0)

    # plotting
    plt.plot(init_test_losses, label='cold start')
    plt.plot(final_test_losses, label=f"learned warm-start k={train_unrolls}")
    plt.plot(prev_sol_losses, label='prev sol')
    plt.plot(nn_losses, label='nearest neighbor')
    plt.yscale('log')
    plt.legend()
    plt.show()
    # import pdb

    # pdb.set_trace()

    # plt.title('losses')
    # plt.plot(train_losses, label='train')
    # init_test_loss = init_test_losses[train_unrolls]
    # final_test_loss = final_test_losses[train_unrolls]
    # test_losses = np.array([init_test_loss, final_test_loss])
    # epochs_array = np.array([0, num_epochs])
    # plt.plot(epochs_array, test_losses, label='test')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.yscale('log')
    # plt.legend()
    # plt.show()

    # assert jnp.linalg.norm(z_stars_test[0, nx:2*nx] - z_stars_test[1, :nx]) <= 1e-3

    # plt.plot(train_losses)
    # plt.yscale('log')
    # # plt.legend()
    # plt.show()
    # import pdb
    # pdb.set_trace()

    # test with osqp in C so that it exactly matches
    #   test with the prev_sol
    # for i in range(traj_length):
    #     osqp_solver.setup(P=P_sparse, q=q, A=A_sparse, l=l, u=u, alpha=1, rho=1, sigma=1, polish=False,
    #                   adaptive_rho=False, scaling=0, max_iter=k, verbose=True, eps_abs=1e-10, eps_rel=1e-10)

    # # fix warm start
    # x_ws = 1*np.ones(n)
    # y_ws = 1*np.ones(m)
    # # s_ws = np.zeros(m)

    # osqp_solver.warm_start(x=x_ws, y=y_ws)
    # results = osqp_solver.solve()
    # z0_mat = prev_z

    # cold-start
    max_iter = 10
    z0_cs = init_eval_out[1][2][:, 0, :]
    osqp_c_out = osqp_model.solve_c(z0_cs, q_mat_test, rel_tol=1e-10,
                                    abs_tol=1e-10, max_iter=max_iter)
    solve_times_cs, solve_iters_cs, x_sols_cs, y_sols_cs = osqp_c_out
    diff_x_cs = x_sols_cs[0, :] - init_eval_out[1][2][0, max_iter, :n]
    diff_y_cs = y_sols_cs[0, :] - init_eval_out[1][2][0, max_iter, n:n + m]
    import pdb
    pdb.set_trace()
    assert jnp.linalg.norm(diff_x_cs) <= 1e-8
    assert jnp.linalg.norm(diff_y_cs) <= 1e-8
    print("=========================================")

    # previous solution
    max_iter = 10
    osqp_c_out = osqp_model.solve_c(prev_z, q_mat_prev, rel_tol=1e-10,
                                    abs_tol=1e-10, max_iter=max_iter)
    solve_times, solve_iters, x_sols, y_sols = osqp_c_out
    diff_x = x_sols[0, :] - prev_sol_out[1][2][0, max_iter, :n]
    diff_y = y_sols[0, :] - prev_sol_out[1][2][0, max_iter, n:n + m]
    assert jnp.linalg.norm(diff_x) <= 1e-8
    assert jnp.linalg.norm(diff_y) <= 1e-8

    
    # assert jnp.linalg.norm(x_jax - results.x) < 1e-10
    # assert jnp.linalg.norm(y_jax - results.y) < 1e-10


# @pytest.mark.skip(reason="temp")
def test_osqp_exact():
    # get a random robust least squares problem
    N_train = 500
    N_test = 50
    N = N_train + N_test
    factor, P, A, q_mat, theta_mat, x_min, x_max, Ad, Bd, rho_vec = multiple_random_mpc_osqp(N,
                                                                                      T=10,
                                                                                      nx=10,
                                                                                      nu=5,
                                                                                      sigma=1,
                                                                                      rho=1,
                                                                                      Ad=None,
                                                                                      Bd=None,
                                                                                      seed=42)
    train_inputs, test_inputs = theta_mat[:N_train, :], theta_mat[N_train:, :]
    z_stars_train, z_stars_test = None, None
    q_mat_train, q_mat_test = q_mat[:N_train, :], q_mat[N_train:, :]
    m, n = A.shape

    c, b = q_mat[0, :n], q_mat[0, n:]
    q, l, u = np.array(q_mat_train[0, :n]), np.array(
        q_mat_train[0, n:n + m]), np.array(q_mat_train[0, n + m:])

    max_iter = 10
    osqp_solver = osqp.OSQP()
    P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))

    osqp_solver.setup(P=P_sparse, q=q, A=A_sparse, l=l, u=u, alpha=1, rho=1, sigma=1, polish=False,
                      adaptive_rho=False, scaling=0, max_iter=max_iter, verbose=True, eps_abs=1e-10, eps_rel=1e-10)

    # fix warm start
    x_ws = np.random.normal(size=(n))
    y_ws = np.random.normal(size=(m))  # 1*np.ones(m)
    # s_ws = np.zeros(m)

    osqp_solver.warm_start(x=x_ws, y=y_ws)
    results = osqp_solver.solve()

    # solve with our jax implementation
    # create the factorization
    sigma = 1
    rho_vec = jnp.ones(m)
    rho_vec = rho_vec.at[l == u].set(1000)

    # M = P + sigma * jnp.eye(n) + rho * A.T @ A
    M = P + sigma * jnp.eye(n) + A.T @ jnp.diag(rho_vec) @ A

    factor = jsp.linalg.lu_factor(M)

    xy0 = jnp.concatenate([x_ws, y_ws])

    z_k, losses, z_all, primal_resids, dual_resids = k_steps_eval_osqp(
        max_iter, xy0, q_mat[0, :], factor, P, A, rho_vec, sigma, supervised=False, z_star=None, jit=True)

    x_jax = z_k[:n]
    y_jax = z_k[n:n + m]
    # data = dict(P=P, A=A, c=c, b=b, cones=cones, x=x_ws, y=y_ws, s=s_ws)
    # sol_hsde = scs_jax(data, hsde=True, iters=max_iters, jit=False,
    #                    rho_x=rho_x, scale=scale, alpha=alpha, plot=False)
    # x_jax, y_jax, s_jax = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    # fp_res_hsde = sol_hsde['fixed_point_residuals']

    # these should match to machine precision
    assert jnp.linalg.norm(x_jax - results.x) < 1e-10
    assert jnp.linalg.norm(y_jax - results.y) < 1e-10
    


@pytest.mark.skip(reason="temp")
def test_random_mpc():
    N_train = 500
    N_test = 50
    N = N_train + N_test
    factor, P, A, q_mat, theta_mat = multiple_random_mpc_osqp(N,
                                                              T=10,
                                                              nx=10,
                                                              nu=5,
                                                              sigma=1,
                                                              rho=1,
                                                              Ad=None,
                                                              Bd=None,
                                                              seed=42)
    train_inputs, test_inputs = theta_mat[:N_train, :], theta_mat[N_train:, :]
    z_stars_train, z_stars_test = None, None
    q_mat_train, q_mat_test = q_mat[:N_train, :], q_mat[N_train:, :]

    # solve the QPs
    z_stars, objvals = solve_many_probs_cvxpy(P, A, q_mat)
    z_stars_train, z_stars_test = z_stars[:N_train, :], z_stars[N_train:, :]

    train_unrolls = 10
    input_dict = dict(algorithm='osqp',
                      q_mat_train=q_mat_train,
                      q_mat_test=q_mat_test,
                      A=A,
                      factor=factor,
                      train_inputs=theta_mat[:N_train, :],
                      test_inputs=theta_mat[N_train:, :],
                      train_unrolls=train_unrolls,
                      nn_cfg={'intermediate_layer_sizes': [300]},
                      jit=True)
    osqp_model = OSQPmodel(input_dict)

    # full evaluation on the test set with nearest neighbor
    k = 1000
    nearest_neighbors_z = get_nearest_neighbors(train_inputs, test_inputs, z_stars_train)
    nn_eval_out = osqp_model.evaluate(k, nearest_neighbors_z,
                                      q_mat_test, z_stars=z_stars_test,
                                      fixed_ws=True, tag='test')
    # nn_z_all = nn_eval_out[1][3]
    # nn_rel_objs = batch_rel_mat(nn_z_all, b_mat_test, objvals_test).mean(axis=1)
    nn_losses = nn_eval_out[1][1].mean(axis=0)

    # full evaluation on the test set
    init_eval_out = osqp_model.evaluate(
        k, test_inputs, q_mat_test, z_stars=z_stars_test, fixed_ws=False, tag='test')
    init_test_losses = init_eval_out[1][1].mean(axis=0)
    # init_z_all = init_eval_out[1][3]

    import pdb
    pdb.set_trace()

    plt.plot(init_test_losses)
    plt.plot(nn_losses)
    plt.yscale('log')
    plt.show()

    # train the osqp_model
    # call train_batch without jitting
    params, state = osqp_model.params, osqp_model.state
    num_epochs = 1000
    train_losses = jnp.zeros(num_epochs)
    for i in range(num_epochs):
        train_result = osqp_model.train_full_batch(params, state)
        loss, params, state = train_result
        train_losses = train_losses.at[i].set(loss)

    osqp_model.params, osqp_model.state = params, state

    # full evaluation on the test set
    # k = 200
    final_eval_out = osqp_model.evaluate(
        k, test_inputs, q_mat_test, z_stars=z_stars_test, fixed_ws=False, tag='test')
    final_test_losses = final_eval_out[1][1].mean(axis=0)
    # final_z_all = init_eval_out[1][3]

    plt.plot(init_test_losses)
    plt.plot(final_test_losses)
    plt.plot(nn_losses)
    plt.yscale('log')
    plt.show()

    plt.plot(train_losses, label='train')
    init_test_loss = init_test_losses[train_unrolls]
    final_test_loss = final_test_losses[train_unrolls]
    test_losses = np.array([init_test_loss, final_test_loss])
    epochs_array = np.array([0, num_epochs])
    plt.plot(epochs_array, test_losses, label='test')
    plt.yscale('log')
    plt.show()

    import pdb
    pdb.set_trace()


@pytest.mark.skip(reason="temp")
def test_basic_osqp():
    m, n = 20, 40
    P = jnp.eye(n)
    c = jnp.array(np.random.normal(size=(n)))
    A = jnp.array(np.random.normal(size=(m, n)))
    l = -jnp.ones(m)
    u = jnp.ones(m)
    rho, sigma = 1, 1

    # create the factorization
    M = P + sigma * jnp.eye(n) + rho * A.T @ A
    factor = jsp.linalg.lu_factor(M)

    # create the vector q
    q = jnp.concatenate([c, l, u])

    k = 1000
    xy0 = jnp.zeros(n + m)

    # z_k, losses = k_steps_train_osqp(k, z0, b, lambd, A, step, supervised=False, z_star=None, jit=True)
    z_k, losses, z_all, primal_resids, dual_resids = k_steps_eval_osqp(
        k, xy0, factor, A, q, rho, sigma, supervised=False, z_star=None, jit=True)
    assert losses[-1] < losses[0] * 1e-6 and losses[0] > .1

    # solve with cvxpy
    x = cp.Variable(n)
    z = cp.Variable(m)
    constraints = [l <= z, z <= u, A @ x == z]
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, P) + c @ x), constraints)
    prob.solve()

    assert jnp.linalg.norm(z_k[n + m:] - z.value) <= 1e-5
    assert jnp.linalg.norm(z_k[:n] - x.value) <= 1e-5


@pytest.mark.skip(reason="temp")
def test_infeas_qp():
    m, n = 11, 10
    P = jnp.eye(n)
    c = jnp.array(np.random.normal(size=(n)))
    A = jnp.array(np.random.normal(size=(m, n)))
    l = jnp.ones(m)
    u = jnp.ones(m)
    rho, sigma = 1, 1

    # create the factorization
    M = P + sigma * jnp.eye(n) + rho * A.T @ A
    factor = jsp.linalg.lu_factor(M)

    # create the vector q
    q = jnp.concatenate([c, l, u])

    k = 1000
    xy0 = jnp.zeros(n + m)

    # z_k, losses = k_steps_train_osqp(k, z0, b, lambd, A, step, supervised=False, z_star=None, jit=True)
    z_k, losses, z_all, primal_resids, dual_resids = k_steps_eval_osqp(
        k, xy0, q, factor, A, rho, sigma, supervised=False, z_star=None, jit=True)
    assert jnp.abs(losses[-1] - losses[-2]) < 1e-6 and losses[0] > .1
    assert losses[-1] > 1e-2


@pytest.mark.skip(reason="temp")
def test_osqp_model():
    N_train = 100
    N_test = 20
    N = N_train + N_test
    factor, A, q_mat, theta_mat = multiple_random_osc_mass_osqp(N)
    train_inputs, test_inputs = theta_mat[:N_train, :], theta_mat[N_train:, :]
    z_stars_train, z_stars_test = None, None
    q_mat_train, q_mat_test = q_mat[:N_train, :], q_mat[N_train:, :]

    input_dict = dict(algorithm='osqp',
                      q_mat_train=q_mat[:N_train, :],
                      q_mat_test=q_mat[N_train:, :],
                      A=A,
                      factor=factor,
                      train_inputs=theta_mat[:N_train, :],
                      test_inputs=theta_mat[N_train:, :],
                      train_unrolls=10)
    osqp_model = OSQPmodel(input_dict)

    # full evaluation on the test set with nearest neighbor
    # nearest_neighbors_z = get_nearest_neighbors(train_inputs, test_inputs, z_stars_train)
    # nn_eval_out = osqp_model.evaluate(500, nearest_neighbors_z,
    #                                   q_mat_test, z_stars=z_stars_test,
    #                                   fixed_ws=True, tag='test')
    # nn_z_all = nn_eval_out[1][3]
    # nn_losses = nn_eval_out[1][1].mean(axis=0)

    # evaluate test before training
    # init_test_loss, init_time_per_iter = l2ws_model.short_test_eval()

    # full evaluation on the test set
    k = 500
    init_eval_out = osqp_model.evaluate(
        k, test_inputs, q_mat_test, z_stars=z_stars_test, fixed_ws=False, tag='test')
    init_test_losses = init_eval_out[1][1].mean(axis=0)
    init_z_all = init_eval_out[1][2]

    plt.plot(init_test_losses)
    plt.yscale('log')
    plt.show()
