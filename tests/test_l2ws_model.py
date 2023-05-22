import time
from examples.robust_ls import random_robust_ls, multiple_random_robust_ls
import jax.numpy as jnp
from l2ws.scs_problem import scs_jax
import scs
import numpy as np
from scipy.sparse import csc_matrix
from l2ws.l2ws_model import L2WSmodel
from l2ws.scs_model import SCSmodel
from l2ws.algo_steps import create_projection_fn, create_M, get_scaled_vec_and_factor
import jax.scipy as jsp
import matplotlib.pyplot as plt


def multiple_random_robust_ls_setup(m_orig, n_orig, rho, b_center, b_range, N_train, N_test, rho_x,
                                    scale):
    N = N_train + N_test
    P, A, cones, q_mat, theta_mat, A_orig, b_orig_mat = multiple_random_robust_ls(
        m_orig, n_orig, rho, b_center, b_range, N)
    m, n = A.shape

    proj = create_projection_fn(cones, n)
    q_mat_train, q_mat_test = q_mat[:N_train, :], q_mat[N_train:N, :]
    train_inputs, test_inputs = theta_mat[:N_train, :], theta_mat[N_train:N, :]
    static_M = create_M(P, A)
    # static_algo_factor = jsp.linalg.lu_factor(static_M + jnp.eye(n + m))
    zero_cone_size = cones['z']
    static_algo_factor, scale_vec = get_scaled_vec_and_factor(
        static_M, rho_x, scale, m, n, zero_cone_size)

    static_prob_data = dict(P=P, A=A, cones=cones, proj=proj,
                            static_M=static_M, static_algo_factor=static_algo_factor,
                            m=m, n=n)
    varying_prob_data = dict(q_mat_train=q_mat_train, q_mat_test=q_mat_test,
                             train_inputs=train_inputs, test_inputs=test_inputs)
    return static_prob_data, varying_prob_data


def test_minimal_l2ws_model():
    """
    tests that we can initialize an L2WSmodel with the minimal amount of information needed

    we test for
    - no errors in creation of L2WSmodel without specifying most the hyperparameters
        such as any part of the neural network
    - no errors during training
    - decrease in training loss after training for 10 epochs of over 50%
    - test loss does not increase by more than 5%
    - warm-starting with our architecture matches the SCS C implementation to machine precision
    """
    # get the problem
    N_train, N_test = 10, 5
    m_orig, n_orig = 30, 40
    rho, b_center, b_range = 1, 1, 1

    # scs hyperparams
    alpha_relax = 1.1
    rho_x = 1.1
    scale = .2

    static_prob_data, varying_prob_data = multiple_random_robust_ls_setup(
        m_orig, n_orig, rho, b_center, b_range, N_train, N_test, rho_x, scale)
    q_mat_train, q_mat_test = varying_prob_data['q_mat_train'], varying_prob_data['q_mat_test']
    train_inputs, test_inputs = varying_prob_data['train_inputs'], varying_prob_data['test_inputs']

    m, n, cones = static_prob_data['m'], static_prob_data['n'], static_prob_data['cones']
    P, A = static_prob_data['P'], static_prob_data['A']
    zero_cone_size = cones['z']

    # enter into the L2WSmodel
    input_dict = dict(algorithm='scs',
                      m=m, n=n, hsde=True, static_flag=True, proj=static_prob_data['proj'],
                      train_unrolls=20, jit=True,
                      q_mat_train=q_mat_train, q_mat_test=q_mat_test,
                      train_inputs=train_inputs, test_inputs=test_inputs,
                      static_M=static_prob_data['static_M'],
                      static_algo_factor=static_prob_data['static_algo_factor'],
                      rho_x=rho_x, scale=scale, alpha_relax=alpha_relax,
                      zero_cone_size=zero_cone_size)
    # l2ws_model = L2WSmodel(input_dict)
    l2ws_model = SCSmodel(input_dict)

    # evaluate test before training
    init_test_loss, init_time_per_iter = l2ws_model.short_test_eval()

    # call train_batch without jitting
    params, state = l2ws_model.params, l2ws_model.state
    num_epochs = 10
    losses = jnp.zeros(num_epochs)
    for i in range(num_epochs):
        train_result = l2ws_model.train_full_batch(params, state)
        loss, params, state = train_result
        losses = losses.at[i].set(loss)

    # some reduction should be made from first to last epoch
    assert losses[0] - losses[-1] > 0

    # final loss should be at least 60% better than the first loss
    assert losses[-1] / losses[0] < 0.6

    l2ws_model.params, l2ws_model.state = params, state

    # evaluate test after training
    final_test_loss, final_time_per_iter = l2ws_model.short_test_eval()

    # test loss does not get 5% worse
    assert final_test_loss < init_test_loss * 1.05

    # after jitting, evaluating the test set should be much faster
    assert final_time_per_iter < .1 * init_time_per_iter

    # evaluate the training set for a different number of iterations
    dynamic_factor, M_dynamic = None, None
    # loss, eval_out, time_per_prob = l2ws_model.evaluate(300, train_inputs, dynamic_factor,
    #                                                     M_dynamic, q_mat_train,
    #                                                     z_stars=None, fixed_ws=False, tag='train')
    loss, eval_out, time_per_prob = l2ws_model.evaluate(300, train_inputs, q_mat_train,
                                                        z_stars=None, fixed_ws=False, tag='train')
    
    # out, losses, iter_losses, angles, primal_residuals, dual_residuals = eval_out
    losses, iter_losses, z_all_plus_1, angles, primal_residuals, dual_residuals, u_all, v_all = eval_out
    # z_all_plus_1, z_final, alpha, u_all, v_all = out

    # warm-start SCS with z0 from all_z_plus_1
    # SCS setup
    max_iters = 6
    P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))
    scs_data = dict(P=P_sparse, A=A_sparse, b=np.zeros(m), c=np.zeros(n))

    solver = scs.SCS(scs_data,
                     cones,
                     normalize=False,
                     scale=scale,
                     adaptive_scale=False,
                     rho_x=rho_x,
                     alpha=alpha_relax,
                     acceleration_lookback=0,
                     max_iters=max_iters,
                     eps_abs=1e-12,
                     eps_rel=0)

    x_ws = z_all_plus_1[0, 0, :n]
    y_ws = z_all_plus_1[0, 0, n:n + m]
    s_ws = np.zeros(m)

    c_np = np.array(q_mat_train[0, :n])
    b_np = np.array(q_mat_train[0, n:])
    solver.update(b=b_np, c=c_np)
    sol = solver.solve(warm_start=True, x=np.array(x_ws), y=np.array(y_ws), s=np.array(s_ws))
    x_c = jnp.array(sol['x'])
    y_c = jnp.array(sol['y'])
    s_c = jnp.array(sol['s'])
    u_final = u_all[0, max_iters - 1, :]
    x_jax = u_final[:n] / u_all[0, max_iters - 1, -1]
    y_jax = u_final[n:n + m] / u_all[0, max_iters - 1, -1]
    s_jax = v_all[0, max_iters - 1, n:n+m] / u_all[0, max_iters - 1, -1]

    import pdb
    pdb.set_trace()

    assert jnp.linalg.norm(x_jax - x_c) < 1e-10
    assert jnp.linalg.norm(y_jax - y_c) < 1e-10
    assert jnp.linalg.norm(s_jax - s_c) < 1e-10
