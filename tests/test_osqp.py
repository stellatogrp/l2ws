import time
import jax.numpy as jnp
import scs
import numpy as np
from scipy.sparse import csc_matrix
from l2ws.algo_steps import k_steps_eval_osqp, k_steps_train_osqp, create_projection_fn, lin_sys_solve, k_steps_train_osqp
import jax.scipy as jsp
import pytest
import matplotlib.pyplot as plt
from l2ws.osqp_model import OSQPmodel
import cvxpy as cp
from jax import vmap
from functools import partial
from examples.osc_mass import multiple_random_osc_mass_osqp
from examples.mpc import multiple_random_mpc_osqp, solve_many_probs_cvxpy
from scipy.spatial import distance_matrix
# from examples.ista import sol_2_obj_diff, solve_many_probs_cvxpy
from l2ws.utils.nn_utils import get_nearest_neighbors
import osqp


# def solve_many_osqp(P, A, q_mat):
#     m, n = A.shape
#     N, p = q_mat.shape
#     for i in range(N):

def test_osqp_exact():
    # get a random robust least squares problem
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
    m, n = A.shape

    c, b = q_mat[0, :n], q_mat[0, n:]
    q, l, u = np.array(q_mat_train[0, :n]), np.array(
        q_mat_train[0, n:n + m]), np.array(q_mat_train[0, n + m:])

    max_iter = 2
    osqp_solver = osqp.OSQP()
    P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))

    
    osqp_solver.setup(P=P_sparse, q=q, A=A_sparse, l=l, u=u, alpha=1, rho=1, sigma=1, polish=False,
                      adaptive_rho=False, scaling=0, max_iter=max_iter, verbose=True, eps_abs=1e-10, eps_rel=1e-10)

    # fix warm start
    x_ws = 0*np.ones(n)
    y_ws = 0*np.ones(m)
    # s_ws = np.zeros(m)

    osqp_solver.warm_start(x=x_ws, y=y_ws)
    results = osqp_solver.solve()
    

    # solve in C
    # P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))
    # c_np, b_np = np.array(c), np.array(b)
    # c_data = dict(P=P_sparse, A=A_sparse, c=c_np, b=b_np)
    # solver = scs.SCS(c_data,
    #                  cones,
    #                  normalize=False,
    #                  scale=scale,
    #                  adaptive_scale=False,
    #                  rho_x=rho_x,
    #                  alpha=alpha,
    #                  acceleration_lookback=0,
    #                  max_iters=max_iters)

    # sol = solver.solve(warm_start=True, x=x_ws, y=y_ws, s=s_ws)
    # x_c = jnp.array(sol['x'])
    # y_c = jnp.array(sol['y'])
    # s_c = jnp.array(sol['s'])

    # solve with our jax implementation
    # create the factorization
    sigma = 1
    rho_vec = jnp.ones(m)
    rho_vec = rho_vec.at[l == u].set(1000)
    
    # M = P + sigma * jnp.eye(n) + rho * A.T @ A
    M = P + sigma * jnp.eye(n) +  A.T @ jnp.diag(rho_vec) @A

    factor = jsp.linalg.lu_factor(M)

    xy0 = jnp.concatenate([x_ws, y_ws])

    z_k, losses, z_all = k_steps_eval_osqp(
        max_iter, xy0, q_mat[0, :], factor, A, rho_vec, sigma, supervised=False, z_star=None, jit=True)
    
    import pdb
    pdb.set_trace()

    # data = dict(P=P, A=A, c=c, b=b, cones=cones, x=x_ws, y=y_ws, s=s_ws)
    # sol_hsde = scs_jax(data, hsde=True, iters=max_iters, jit=False,
    #                    rho_x=rho_x, scale=scale, alpha=alpha, plot=False)
    # x_jax, y_jax, s_jax = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    # fp_res_hsde = sol_hsde['fixed_point_residuals']

    # these should match to machine precision
    # assert jnp.linalg.norm(x_jax - x_c) < 1e-10
    # assert jnp.linalg.norm(y_jax - y_c) < 1e-10
    # assert jnp.linalg.norm(s_jax - s_c) < 1e-10


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
    z_k, losses, z_all = k_steps_eval_osqp(
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
    z_k, losses, z_all = k_steps_eval_osqp(
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
    init_z_all = init_eval_out[1][3]

    plt.plot(init_test_losses)
    plt.yscale('log')
    plt.show()
