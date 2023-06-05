import time
from examples.robust_ls import random_robust_ls
from examples.sparse_pca import multiple_random_sparse_pca
from examples.robust_kalman import multiple_random_robust_kalman
import jax.numpy as jnp
from l2ws.scs_problem import scs_jax
import scs
import numpy as np
from scipy.sparse import csc_matrix
from l2ws.algo_steps import k_steps_eval_scs, k_steps_train_scs, create_projection_fn, lin_sys_solve, \
    create_M, get_scale_vec, get_scaled_factor
import jax.scipy as jsp
import pytest
from matplotlib import pyplot as plt


def test_train_vs_eval():
    # get a random robust least squares problem
    m_orig, n_orig = 20, 25
    rho = 1
    b_center, b_range = 1, 1
    P, A, c, b, cones = random_robust_ls(m_orig, n_orig, rho, b_center, b_range)
    m, n = A.shape
    zero_cone_size = cones['z']
    proj = create_projection_fn(cones, n)
    k = 20
    z0 = jnp.ones(m + n + 1)
    M = create_M(P, A)

    rho_x, scale = 1e-5, .1
    scale_vec = get_scale_vec(rho_x, scale, m, n, zero_cone_size)
    scale_vec_diag = jnp.diag(scale_vec)
    factor = jsp.linalg.lu_factor(M + scale_vec_diag)

    q = jnp.concatenate([c, b])
    q_r = lin_sys_solve(factor, q)

    train_out = k_steps_train_scs(k, z0, q_r, factor, supervised=False,
                              z_star=None, proj=proj, jit=False, hsde=True,
                              m=m, n=n, zero_cone_size=zero_cone_size, rho_x=rho_x, scale=scale)
    z_final_train, iter_losses_train = train_out

    eval_out = k_steps_eval_scs(k, z0, q_r, factor, proj, P, A, c, b, jit=True,
                            hsde=True, zero_cone_size=zero_cone_size, rho_x=rho_x, scale=scale)
    z_final_eval, iter_losses_eval = eval_out[:2]
    assert jnp.linalg.norm(iter_losses_train - iter_losses_eval) <= 1e-10
    assert jnp.linalg.norm(z_final_eval - z_final_train) <= 1e-10


def test_jit_speed():
    # problem setup
    m_orig, n_orig = 30, 40
    rho = 1
    b_center, b_range = 1, 1
    P, A, c, b, cones = random_robust_ls(m_orig, n_orig, rho, b_center, b_range)
    m, n = A.shape
    max_iters = 1000

    # fix warm start
    x_ws = np.ones(n)
    y_ws = np.ones(m)
    s_ws = np.zeros(m)

    # solve with jit
    t0_jit = time.time()
    data = dict(P=P, A=A, c=c, b=b, cones=cones, x=x_ws, y=y_ws, s=s_ws)
    sol_hsde = scs_jax(data, hsde=True, iters=max_iters)
    x_jit, y_jit, s_jit = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    fp_res_jit = sol_hsde['fixed_point_residuals']
    t1_jit = time.time()
    jit_time = t1_jit - t0_jit

    # solve without jit
    t0_non_jit = time.time()
    data = dict(P=P, A=A, c=c, b=b, cones=cones, x=x_ws, y=y_ws, s=s_ws)
    sol_hsde = scs_jax(data, hsde=True, iters=max_iters)
    x_non_jit, y_non_jit, s_non_jit = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    fp_res_non_jit = sol_hsde['fixed_point_residuals']
    t1_non_jit = time.time()
    non_jit_time = t1_non_jit - t0_non_jit

    assert jit_time - non_jit_time > 0
    # assert jnp.all(jnp.diff(fp_res_jit) < 1e-10)
    # assert jnp.all(jnp.diff(fp_res_non_jit) < 1e-10)

    # these should match to machine precision
    assert jnp.linalg.norm(x_jit - x_non_jit) < 1e-10
    assert jnp.linalg.norm(y_jit - y_non_jit) < 1e-10
    assert jnp.linalg.norm(s_jit - s_non_jit) < 1e-10

    # make sure the residuals start high and end very low
    assert fp_res_jit[0] > .1 and fp_res_non_jit[0] > .1
    assert fp_res_jit[-1] < 1e-6 and fp_res_non_jit[-1] > 1e-16
    assert fp_res_jit[-1] < 1e-8 and fp_res_non_jit[-1] > 1e-16


def test_hsde_socp_robust_ls():
    """
    tests to make sure hsde returns the same solution as the non-hsde
    tests socp of different cone sizes also (there are 2 SOCs)
    """
    # get a random robust least squares problem
    m_orig, n_orig = 50, 55
    rho = 1
    b_center, b_range = 1, 1
    P, A, c, b, cones = random_robust_ls(m_orig, n_orig, rho, b_center, b_range)

    data = dict(P=P, A=A, c=c, b=b, cones=cones)
    iters = 400

    sol_std = scs_jax(data, hsde=False, iters=iters, rho_x=1, scale=1, alpha=1)
    x_std, y_std, s_std = sol_std['x'], sol_std['y'], sol_std['s']
    fp_res_std = sol_std['fixed_point_residuals']

    sol_hsde = scs_jax(data, hsde=True, iters=iters)
    x_hsde, y_hsde, s_hsde = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    fp_res_hsde = sol_hsde['fixed_point_residuals']

    # import pdb
    # pdb.set_trace()

    assert jnp.linalg.norm(x_hsde - x_std) < 1e-3
    assert jnp.linalg.norm(y_hsde - y_std) < 1e-3
    assert jnp.linalg.norm(s_hsde - s_std) < 1e-3

    # make sure the residuals start high and end very low
    assert fp_res_std[0] > .1 and fp_res_std[0] > .1
    assert fp_res_std[-1] < 1e-4 and fp_res_std[-1] > 1e-16
    assert fp_res_hsde[-1] < 1e-4 and fp_res_hsde[-1] > 1e-16


def test_c_socp_robust_kalman_filter_relaxation():
    """
    tests to make sure hsde returns the same solution as the non-hsde
    tests socp of different cone sizes also (there are 2 SOCs)
    """
    # get a random robust least squares problem
    P, A, cones, q_mat, theta_mat = multiple_random_robust_kalman(
        N=5, T=50, gamma=.05, dt=.5, mu=2, rho=2, sigma=20, p=0, w_noise_var=.1, y_noise_var=.1)
    m, n = A.shape

    c, b = q_mat[0, :n], q_mat[0, n:]
    data = dict(P=P, A=A, c=c, b=b, cones=cones)

    # sol_std = scs_jax(data, hsde=False, iters=200)
    # x_std, y_std, s_std = sol_std['x'], sol_std['y'], sol_std['s']
    # fp_res_std = sol_std['fixed_point_residuals']

    # sol_hsde = scs_jax(data, hsde=True, iters=200)
    # x_hsde, y_hsde, s_hsde = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    # fp_res_hsde = sol_hsde['fixed_point_residuals']

    # fix warm start
    x_ws = np.ones(n)
    y_ws = np.ones(m)
    s_ws = np.zeros(m)
    max_iters = 30

    # pick algorithm hyperparameters
    rho_x = 1
    scale = 1
    alpha = 1

    # solve in C
    P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))
    c_np, b_np = np.array(c), np.array(b)
    c_data = dict(P=P_sparse, A=A_sparse, c=c_np, b=b_np)
    solver = scs.SCS(c_data,
                     cones,
                     normalize=False,
                     scale=scale,
                     adaptive_scale=False,
                     rho_x=rho_x,
                     alpha=alpha,
                     acceleration_lookback=0,
                     max_iters=max_iters)

    sol = solver.solve(warm_start=True, x=x_ws, y=y_ws, s=s_ws)
    x_c = jnp.array(sol['x'])
    y_c = jnp.array(sol['y'])
    s_c = jnp.array(sol['s'])

    # solve with our jax implementation
    data = dict(P=P, A=A, c=c, b=b, cones=cones, x=x_ws, y=y_ws, s=s_ws)
    sol_hsde = scs_jax(data, hsde=True, iters=max_iters, jit=False,
                       rho_x=rho_x, scale=scale, alpha=alpha, plot=False)
    x_jax, y_jax, s_jax = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    fp_res_hsde = sol_hsde['fixed_point_residuals']

    # these should match to machine precision
    assert jnp.linalg.norm(x_jax - x_c) < 1e-10
    assert jnp.linalg.norm(y_jax - y_c) < 1e-10
    assert jnp.linalg.norm(s_jax - s_c) < 1e-10

    # make sure the residuals start high and end very low
    assert fp_res_hsde[0] > 10
    assert fp_res_hsde[-1] < .5 and fp_res_hsde[-1] > 1e-16

    import pdb
    pdb.set_trace()
    


def test_c_vs_jax_sdp():
    """
    check iterate returned by x vs one returned by jax with warm-start is the same
    """
    # get a random sparse pca problem
    P, A, cones, q_mat, theta_mat_jax, A_tensor = multiple_random_sparse_pca(
        n_orig=30, k=10, r=10, N=5)
    m, n = A.shape

    max_iters = 10

    # fix warm start
    x_ws = np.ones(n)
    y_ws = np.ones(m)
    s_ws = np.zeros(m)

    # solve in C
    c, b = q_mat[0, :n], q_mat[0, n:]
    P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))
    c_np, b_np = np.array(c), np.array(b)
    c_data = dict(P=P_sparse, A=A_sparse, c=c_np, b=b_np)
    solver = scs.SCS(c_data,
                     cones,
                     normalize=False,
                     scale=.1,
                     adaptive_scale=False,
                     rho_x=.01,
                     alpha=1.6,
                     acceleration_lookback=0,
                     max_iters=max_iters)

    sol = solver.solve(warm_start=True, x=x_ws, y=y_ws, s=s_ws)
    x_c = jnp.array(sol['x'])
    y_c = jnp.array(sol['y'])
    s_c = jnp.array(sol['s'])

    # solve with our jax implementation
    data = dict(P=P, A=A, c=c, b=b, cones=cones, x=x_ws, y=y_ws, s=s_ws)
    sol_hsde = scs_jax(data, hsde=True, iters=max_iters, alpha=1.6, scale=.1, rho_x=.01)
    x_jax, y_jax, s_jax = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    # fp_res_hsde = sol_hsde['fixed_point_residuals']

    # these should match to machine precision
    assert jnp.linalg.norm(x_jax - x_c) < 1e-6
    assert jnp.linalg.norm(y_jax - y_c) < 1e-6
    assert jnp.linalg.norm(s_jax - s_c) < 1e-6
    # assert jnp.all(jnp.diff(fp_res_hsde) < 0)


def test_c_vs_jax_socp():
    """
    check iterate returned by x vs one returned by jax with war-start is the same
    """
    # get a random robust least squares problem
    m_orig, n_orig = 30, 40
    rho = 1
    b_center, b_range = 1, 1
    P, A, c, b, cones = random_robust_ls(m_orig, n_orig, rho, b_center, b_range)
    m, n = A.shape

    max_iters = 10

    # fix warm start
    x_ws = np.ones(n)
    y_ws = np.ones(m)
    s_ws = np.zeros(m)

    # select hyperparameters
    scale = 10
    rho_x = 1e-3
    alpha = 1.8

    # solve in C
    P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))
    c_np, b_np = np.array(c), np.array(b)
    c_data = dict(P=P_sparse, A=A_sparse, c=c_np, b=b_np)
    solver = scs.SCS(c_data,
                     cones,
                     normalize=False,
                     scale=scale,
                     adaptive_scale=False,
                     rho_x=rho_x,
                     alpha=alpha,
                     acceleration_lookback=0,
                     max_iters=max_iters)

    sol = solver.solve(warm_start=True, x=x_ws, y=y_ws, s=s_ws)
    x_c = jnp.array(sol['x'])
    y_c = jnp.array(sol['y'])
    s_c = jnp.array(sol['s'])

    # solve with our jax implementation
    data = dict(P=P, A=A, c=c, b=b, cones=cones, x=x_ws, y=y_ws, s=s_ws)
    sol_hsde = scs_jax(data, hsde=True, iters=max_iters, scale=scale, rho_x=rho_x, alpha=alpha)
    x_jax, y_jax, s_jax = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    fp_res_hsde = sol_hsde['fixed_point_residuals']

    # these should match to machine precision
    assert jnp.linalg.norm(x_jax - x_c) < 1e-10
    assert jnp.linalg.norm(y_jax - y_c) < 1e-10
    assert jnp.linalg.norm(s_jax - s_c) < 1e-10
    assert jnp.all(jnp.diff(fp_res_hsde) < 0)


def test_warm_start_from_opt():
    """
    this is the only test that uses a different warm-start from zero for s
    it's important for the non-identiy DR scaling
    """
    m_orig, n_orig = 30, 40
    rho = 1
    b_center, b_range = 1, 1
    P, A, c, b, cones = random_robust_ls(m_orig, n_orig, rho, b_center, b_range)
    m, n = A.shape

    max_iters = 1

    # fix warm start
    x_ws = np.ones(n)
    y_ws = np.ones(m)
    s_ws = np.zeros(m)

    # solve in C to get close to opt
    P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))
    c_np, b_np = np.array(c), np.array(b)
    c_data = dict(P=P_sparse, A=A_sparse, c=c_np, b=b_np)
    solver_opt = scs.SCS(c_data,
                         cones,
                         normalize=False,
                         scale=1,
                         adaptive_scale=False,
                         rho_x=1,
                         acceleration_lookback=0,
                         max_iters=1000)

    sol = solver_opt.solve(warm_start=True, x=x_ws, y=y_ws, s=s_ws)
    x_opt = jnp.array(sol['x'])
    y_opt = jnp.array(sol['y'])
    s_opt = jnp.array(sol['s'])

    # set hyperparameters
    rho_x = .1
    alpha = 1.5
    scale = 1.01

    # warm start scs from opt
    solver = scs.SCS(c_data,
                     cones,
                     normalize=False,
                     scale=scale,
                     adaptive_scale=False,
                     rho_x=rho_x,
                     alpha=alpha,
                     acceleration_lookback=0,
                     max_iters=max_iters,
                     eps_abs=1e-12,
                     eps_rel=1e-12,)
    sol = solver.solve(warm_start=True, x=np.array(x_opt), y=np.array(y_opt), s=np.array(s_opt))
    x_c = jnp.array(sol['x'])
    y_c = jnp.array(sol['y'])
    s_c = jnp.array(sol['s'])

    # warm start our implementation from opt
    data = dict(P=P, A=A, c=c, b=b, cones=cones, x=x_opt, y=y_opt, s=s_opt)
    sol_hsde = scs_jax(data, hsde=True, jit=False, iters=max_iters,
                       rho_x=rho_x, scale=scale, alpha=alpha)
    x_jax, y_jax, s_jax = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    fp_res_hsde = sol_hsde['fixed_point_residuals']

    # these should match to machine precision
    assert jnp.linalg.norm(x_jax - x_c) < 1e-12
    assert jnp.linalg.norm(y_jax - y_c) < 1e-12
    assert jnp.linalg.norm(s_jax - s_c) < 1e-12

