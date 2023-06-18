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
# from examples.osc_mass import multiple_random_osc_mass_osqp
from examples.mpc import multiple_random_mpc_osqp, solve_many_probs_cvxpy, solve_multiple_trajectories, shifted_sol
# from examples.mnist import get_mnist, vectorized2DBlurMatrix, mnist_canon
from scipy.spatial import distance_matrix
# from examples.ista import sol_2_obj_diff, solve_many_probs_cvxpy
from l2ws.utils.nn_utils import get_nearest_neighbors
import osqp
from l2ws.utils.mpc_utils import closed_loop_rollout
from examples.quadcopter import quadcopter_dynamics, QUADCOPTER_NX, QUADCOPTER_NU, plot_traj_3d


def test_plot():
    pass


def test_basic_rollout():
    """
    tests a basic rollout with zero control
    """
    
    
    sim_len = 40
    budget = 5
    dynamics = quadcopter_dynamics
    T = 10
    nx, nu = QUADCOPTER_NX, QUADCOPTER_NU
    system_constants = dict(T=T, nx=nx, nu=nu, dt=.01)
    x_init_traj = jnp.zeros(nx)
    traj_list = [x_init_traj for i in range(sim_len)] 

    # placeholder qp solver 
    n = T * nx
    m = T * nx
    def qp_solver(A, B, x0, u0, ref_traj, budget):
        return jnp.zeros(n + 2 * m)
    sols, state_traj_list = closed_loop_rollout(qp_solver, x_init_traj, dynamics, system_constants, traj_list, budget, noise_list=None)
    import pdb
    pdb.set_trace()
    # state_traj_list = 
    plot_traj_3d([state_traj_list], labels=['blank'])


@pytest.mark.skip(reason="temp")
def test_quadcopter():
    N_train = 1000
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

    train_unrolls = 5
    input_dict = dict(rho=rho_vec,
                      q_mat_train=q_mat_train,
                      q_mat_test=q_mat_test,
                      P=P,
                      A=A,
                      factor=factor,
                      train_inputs=theta_mat[:N_train, :],
                      test_inputs=theta_mat[N_train:, :],
                      train_unrolls=train_unrolls,
                      nn_cfg={'intermediate_layer_sizes': [100, 100]},
                      supervised=True,
                      z_stars_train=z_stars_train,
                      z_stars_test=z_stars_test,
                      jit=True)
    osqp_model = OSQPmodel(input_dict)

    sim_len = 20
    q_init = q_mat_test[0, :]
    x_init = theta_mat_test[0, :]
    k_plot = 30
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
    num_epochs = 2000
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




# def simulate_fwd_l2ws(sim_len, l2ws_model, k, noise_vec_list, q_init, x_init, A, Ad, Bd, T, nx, nu, prev_sol=False):
#     """
#     does the forward simulation

#     returns
#     """
#     m, n = A.shape
#     # get the first test_input and q_mat_test
#     input = x_init
#     q_mat = q_init

#     opt_sols = []
#     state_traj = [x_init]

#     opt_sol = np.zeros(n + 2 * m)
    

#     for i in range(sim_len):
#         # evaluate
#         if prev_sol:
#             # get the shifted previous solution
#             prev_z_shift = shifted_sol(opt_sol[:m + n], T, nx, nu, m, n)
#             final_eval_out = l2ws_model.evaluate(
#                 k, prev_z_shift[None, :], q_mat[None, :], z_stars=None, fixed_ws=True, tag='test')
#             # z_star = final_eval_out[1][2][0, -1, :]
#         else:
#             final_eval_out = l2ws_model.evaluate(
#                 k, input[None, :], q_mat[None, :], z_stars=None, fixed_ws=False, tag='test')
#         print('loss', k, prev_sol, final_eval_out[1][0])

#         # get the first control input
#         # final_eval_out[1][2] will have shape (1, k, n + 2 * m)
#         opt_sol = final_eval_out[1][2][0, -1, :]

#         u0 = opt_sol[T * nx: T * nx + nu]

#         # import pdb
#         # pdb.set_trace()

#         # input the first control to get the next state and perturb it
#         x_init = Ad @ x_init + Bd @ u0 + noise_vec_list[i]

#         # set test_input and q_mat_test
#         input = x_init
#         c, l, u = q_mat[:n], q_mat[n:n + m], q_mat[n + m:]
#         Ad_x_init = Ad @ x_init
#         l = l.at[:nx].set(-Ad_x_init)
#         u = u.at[:nx].set(-Ad_x_init)
#         q_mat = q_mat.at[n:n + m].set(l)
#         q_mat = q_mat.at[n + m:].set(u)

#         # append to the optimal solutions
#         opt_sols.append(opt_sol)

#         # append to the state trajectory
#         state_traj.append(x_init)

#     return opt_sols, state_traj
