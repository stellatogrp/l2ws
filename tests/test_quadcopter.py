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
from l2ws.utils.mpc_utils import closed_loop_rollout, static_canon_mpc_osqp
from examples.quadcopter import quadcopter_dynamics, QUADCOPTER_NX, QUADCOPTER_NU, plot_traj_3d, make_obstacle_course, YPRToQuat



# @pytest.mark.skip(reason="temp")
def test_optimal_control_obstacle_rollout():
    """
    runs the optimal controller for a large number of steps to get from point A to point B
    point A: origin
    point B: pos = (2, 2, 1), velocity = (0, 0, 0)
    """
    # get the dynamics and set other basic parameters
    sim_len = 500
    budget = 5
    dynamics = quadcopter_dynamics
    T = 20
    nx, nu = QUADCOPTER_NX, QUADCOPTER_NU
    dt = .05
    
    x_init_traj = jnp.zeros(nx)

    # x_init_traj = x_init_traj.at[6].set(1)
    # quaternion = YPRToQuat(jnp.zeros(3))
    # x_init_traj = x_init_traj.at[6:10].set(quaternion)

    # create the parametric problems
    N_train = 100
    N_test = 100
    N = N_train + N_test

    num_traj = int(N_test / sim_len)
    num_traj_train = int(N_train / sim_len)
    x_init_factor = 10
    noise_std_dev = .00

    # set (x_min, x_max, u_min, u_max)
    x_bds = np.inf
    angle_bds = .5
    vel_bds = np.inf
    omega_bds = 1
    x_min = np.array([-x_bds,-x_bds,-x_bds,-vel_bds,-vel_bds,-vel_bds,
                    -angle_bds,-angle_bds,-np.inf,-omega_bds,-omega_bds,-omega_bds])
    x_max = np.array([ x_bds, x_bds, x_bds, vel_bds, vel_bds, vel_bds,
                    angle_bds, angle_bds, np.inf, omega_bds, omega_bds, omega_bds])
    # x_min = np.array([-x_bds,-x_bds,-x_bds,-vel_bds,-vel_bds,-vel_bds,
    #                 -np.inf, -np.inf, -np.inf, -np.inf])
    # x_max = np.array([x_bds, x_bds, x_bds, vel_bds, vel_bds, vel_bds,
    #                 np.inf, np.inf, np.inf, np.inf])
    # x_min = np.array([-x_bds,-x_bds,-x_bds,-vel_bds,-vel_bds,-vel_bds,
    #                 -angle_bds,-angle_bds,-np.inf,-omega_bds,-omega_bds,-omega_bds])
    # x_max = np.array([ x_bds, x_bds, x_bds, vel_bds, vel_bds, vel_bds,
    #                 angle_bds, angle_bds, np.inf, omega_bds, omega_bds, omega_bds])
    # u0 = 10.5916
    # u_min = np.array([9.6, 9.6, 9.6, 9.6]) - u0
    # u_max = np.array([13., 13., 13., 13.]) - u0
    # u_min = np.zeros(nu)
    # u_min[4:] = -10
    # u_max = 10 * np.ones(nu) #np.array([np.inf, np.inf, np.inf, np.inf])
    # u_min = -u_max
    # Quadrotor constant
    roll_max = 6.0
    pitch_max = 6.0
    yaw_max = 6.0
    thrust_min = 2.0
    thrust_max = 20.0
    # u_max = np.array([thrust_max, roll_max, pitch_max, yaw_max])
    # u_min = np.array([thrust_min, -roll_max, -pitch_max, -yaw_max])
    u_max = np.array([10, 10, 10, 10])
    u_min = -u_max

    delta_u = np.array([20, 6, 6, 6])
    # u_min = -u_max
    system_constants = dict(T=T, nx=nx, nu=nu, dt=dt, 
                            u_min=u_min, u_max=u_max,
                            x_min=x_min, x_max=x_max)

    # set (Q_t, Q_T, R)
    # Q = jnp.diag(jnp.array([100, 100, 100, 
    #                         100, 100, 100, 
    #                         0, 0, 0, 0])) 
    Q = jnp.diag(jnp.array([100, 100, 100, 
                            100, 100, 100, 
                            0, 0, 0, 0, 0, 0])) 
    QT = 10 * Q #jnp.diag(jnp.array([1, 1, 1, .0, .0, 0, 0, 0, 0, 0]))
    R = 0.1 * jnp.eye(nu)

    # reference trajectory dict
    traj_list = make_obstacle_course()
    # Q_ref = jnp.diag(jnp.array([1, 1, 1, 
    #                         0, 0, 0, 
    #                         0, 0, 0, 0]))
    Q_ref = jnp.diag(jnp.array([1, 1, 1, 
                            0, 0, 0, 
                            0, 0, 0, 0, 0, 0]))
    ref_traj_dict = dict(case='obstacle_course', traj_list=traj_list, Q=Q_ref, tol=.05)

    # run the optimal controller to get everything using closed_loop_rollout
    #   1. (P, A, factor) for each problem
    #   2. (c, b) for each problem
    #   3. theta = (x0, u0, ref_traj)
    cd0 = jnp.zeros(nx)
    cd0 = cd0.at[5].set(-9.8 * dt) * 0
    u_prev = np.array([9.8, 0, 0, 0])
    static_canon_mpc_osqp_partial = partial(static_canon_mpc_osqp, T=T, nx=nx, nu=nu, 
                                            x_min=x_min, x_max=x_max, u_min=u_min, 
                                            u_max=u_max, Q=Q, QT=QT, R=R, delta_u=delta_u)

    # create the opt_qp_solver
    # def opt_qp_solver(Ad, Bd, x0, u0, ref_traj, budget, prev_sol):
    u00 = jnp.array([9.8, 0, 0, 0])
    def opt_qp_solver(Ac, Bc, x0, u0, x_dot, ref_traj, budget, prev_sol):
        # get the discrete time system Ad, Bd from the continuous time system Ac, Bc
        Ad = jnp.eye(nx) + Ac * dt
        Bd = Bc * dt

        # print('Ad', Ad)
        # print('Bd', Bd)

        # no need to use u0 for the non-learned case

        # get the constants for the discrete system
        cd = cd0 +  (x_dot - Ac @ x0 - Bc @ u0) * dt
        # cd = cd0 - (Ac @ x0 + Bc @ u0) * dt
        print('cd', cd)

        # get (P, A, c, l, u)
        # out_dict = static_canon_mpc_osqp_partial(ref_traj, x0*dt, Ad, Bd, cd=cd, u_prev=u0)
        out_dict = static_canon_mpc_osqp_partial(ref_traj, x0, Ad, Bd, cd=cd, u_prev=u0)
        P, A, c, l, u = out_dict['P'], out_dict['A'], out_dict['c'], out_dict['l'], out_dict['u']
        m, n = A.shape
        q = jnp.concatenate([c, l, u])

        # get factor
        rho_vec, sigma = jnp.ones(m), 1
        rho_vec = rho_vec.at[l == u].set(1000)
        M = P + sigma * jnp.eye(n) + A.T @ jnp.diag(rho_vec) @ A
        factor = jsp.linalg.lu_factor(M)

        # solve
        # import pdb
        # pdb.set_trace()
        z0 = prev_sol #jnp.zeros(m + n)
        # z0 = jnp.zeros(m + n)
        out = k_steps_eval_osqp(budget, z0, q, factor, P, A, rho=rho_vec, 
                                sigma=sigma, supervised=False, z_star=None, jit=True)
        sol = out[0]
        print('loss', out[1][-1])
        # plt.plot(out[1])
        # plt.yscale('log')
        # plt.show()
        # plt.clf()
        
        # z0 = sol[:nx]
        # w0 = sol[T*nx:T*nx + nu]
        # z1 = sol[nx:2*nx]
        # w1 = sol[T*nx + nu:T*nx + 2*nu]
        # import pdb
        # pdb.set_trace()
        
        return sol, P, A, factor, q
    
    # do the closed loop rollout
    opt_budget = 10
    rollout_results = closed_loop_rollout(opt_qp_solver, sim_len, x_init_traj, dynamics, 
                                          system_constants, ref_traj_dict, opt_budget, noise_list=None)
    state_traj_list = rollout_results[0]
    print('state_traj_list', state_traj_list)
    plot_traj_3d([state_traj_list], goals=traj_list, labels=['blank'])

    # assert jnp.linalg.norm(state_traj_list[20][:2]) == 0
    # assert state_traj_list[20][2] < -.1

    # mpc_setup = multiple_random_mpc_osqp(N_train,
    #                                      T=T,
    #                                      nx=nx,
    #                                      nu=nu,
    #                                      Ad=None,
    #                                      Bd=None,
    #                                      seed=42,
    #                                      x_init_factor=x_init_factor,
    #                                      quadcopter=True)
    # factor, P, A, q_mat_train, theta_mat_train, x_min, x_max, Ad, Bd, rho_vec = mpc_setup
    # m, n = A.shape
    # q = q_mat_train[0, :]

    # theta_mat_train, z_stars_train, q_mat_train = solve_multiple_trajectories(
    #     sim_len, num_traj_train, x_min, x_max, x_init_factor, Ad, P, A, q, noise_std_dev)

    # theta_mat_test, z_stars_test, q_mat_test = solve_multiple_trajectories(
    #     sim_len, num_traj, x_min, x_max, x_init_factor, Ad, P, A, q, noise_std_dev)

    # # create theta_mat and q_mat
    # q_mat = jnp.vstack([q_mat_train, q_mat_test])
    # theta_mat = jnp.vstack([theta_mat_train, theta_mat_test])
    # train_unrolls = 5
    # input_dict = dict(rho=rho_vec,
    #                 #   q_mat_train=q_mat_train,
    #                 #   q_mat_test=q_mat_test,
    #                   m=m,
    #                   n=n,
    #                 #   P=P,
    #                 #   A=A,
    #                 #   factor=factor,
    #                   train_inputs=theta_mat[:N_train, :],
    #                   test_inputs=theta_mat[N_train:, :],
    #                   train_unrolls=train_unrolls,
    #                   nn_cfg={'intermediate_layer_sizes': [100, 100]},
    #                   supervised=True,
    #                   z_stars_train=z_stars_train,
    #                   z_stars_test=z_stars_test,
    #                   jit=True)
    # # osqp_model = OSQPmodel(input_dict)

    # # create the qp_solver from the OSQPmodel
    # def qp_solver(Ad, Bd, x0, u0, ref_traj, budget):
    #     # get (P, A, c, l, u)

    #     # get factor

    #     # osqp_model.evaluate
    #     sol = osqp_model.eval(budget)
    #     return sol, P, A, factor, q

    # # do the closed loop rollout
    # sols, state_traj_list = closed_loop_rollout(qp_solver, sim_len, x_init_traj, dynamics, system_constants, ref_traj_dict, budget, noise_list=None)
    # assert jnp.linalg.norm(state_traj_list[20][:2]) == 0
    # assert state_traj_list[20][2] < -.1


@pytest.mark.skip(reason="temp")
def test_zero_control_rollout():
    """
    tests a basic rollout with zero control

    the quadcopter should just drop vertically
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
    def qp_solver(Ad, Bd, x0, u0, ref_traj, budget):
        sol = jnp.zeros(n + 2 * m)
        P = jnp.zeros((n, n))
        A = jnp.zeros((m, n))
        factor = None
        q = jnp.zeros(n + 2 * m)
        return sol, P, A, factor, q
    
    # reference trajectory dict
    ref_traj_dict = dict(case='fixed_path', traj_list=traj_list)

    # do the closed loop rollout
    rollout_results = closed_loop_rollout(qp_solver, sim_len, x_init_traj, dynamics, system_constants, ref_traj_dict, budget, noise_list=None)
    state_traj_list = rollout_results[0]

    # assert that the position moves down vertically
    assert jnp.linalg.norm(state_traj_list[20][:2]) == 0
    assert state_traj_list[20][2] < -.1

    # plot should show vertical drop that accelerates
    # plot_traj_3d([state_traj_list], labels=['blank'])


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
