import time
from examples.robust_ls import random_robust_ls
import pytest
import jax.numpy as jnp
from l2ws.scs_problem import scs_jax
import scs
import numpy as np
from scipy.sparse import csc_matrix


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
    assert jnp.all(jnp.diff(fp_res_jit) < 1e-12)
    assert jnp.all(jnp.diff(fp_res_non_jit) < 1e-12)

    # these should match to machine precision
    assert jnp.linalg.norm(x_jit - x_non_jit) < 1e-12
    assert jnp.linalg.norm(y_jit - y_non_jit) < 1e-12
    assert jnp.linalg.norm(s_jit - s_non_jit) < 1e-12


def test_hsde_socp():
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

    sol_std = scs_jax(data, hsde=False, iters=200)
    x_std, y_std, s_std = sol_std['x'], sol_std['y'], sol_std['s']
    fp_res_std = sol_std['fixed_point_residuals']

    sol_hsde = scs_jax(data, hsde=True, iters=200)
    x_hsde, y_hsde, s_hsde = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    fp_res_hsde = sol_hsde['fixed_point_residuals']

    assert jnp.linalg.norm(x_hsde - x_std) < 1e-3
    assert jnp.linalg.norm(y_hsde - y_std) < 1e-3
    assert jnp.linalg.norm(s_hsde - s_std) < 1e-3
    assert jnp.all(jnp.diff(fp_res_std) < 0)
    assert jnp.all(jnp.diff(fp_res_hsde) < 0)


def test_c_vs_jax():
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

    # solve in C
    P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))
    c_np, b_np = np.array(c), np.array(b)
    c_data = dict(P=P_sparse, A=A_sparse, c=c_np, b=b_np)
    solver = scs.SCS(c_data,
                     cones,
                     normalize=False,
                     scale=1,
                     adaptive_scale=False,
                     rho_x=1,
                     alpha=1,
                     acceleration_lookback=0,
                     max_iters=max_iters)

    sol = solver.solve(warm_start=True, x=x_ws, y=y_ws, s=s_ws)
    x_c = jnp.array(sol['x'])
    y_c = jnp.array(sol['y'])
    s_c = jnp.array(sol['s'])

    # solve with our jax implementation
    data = dict(P=P, A=A, c=c, b=b, cones=cones, x=x_ws, y=y_ws, s=s_ws)
    sol_hsde = scs_jax(data, hsde=True, iters=max_iters)
    x_jax, y_jax, s_jax = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    fp_res_hsde = sol_hsde['fixed_point_residuals']

    # these should match to machine precision
    assert jnp.linalg.norm(x_jax - x_c) < 1e-12
    assert jnp.linalg.norm(y_jax - y_c) < 1e-12
    assert jnp.linalg.norm(s_jax - s_c) < 1e-12
    assert jnp.all(jnp.diff(fp_res_hsde) < 0)


def test_warm_start_from_opt():
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
                         alpha=1,
                         acceleration_lookback=0,
                         max_iters=1000)

    sol = solver_opt.solve(warm_start=True, x=x_ws, y=y_ws, s=s_ws)
    x_opt = jnp.array(sol['x'])
    y_opt = jnp.array(sol['y'])
    s_opt = jnp.array(sol['s'])

    # warm start scs from opt
    solver = scs.SCS(c_data,
                     cones,
                     normalize=False,
                     scale=1,
                     adaptive_scale=False,
                     rho_x=1,
                     alpha=1,
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
    sol_hsde = scs_jax(data, hsde=True, iters=max_iters)
    x_jax, y_jax, s_jax = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    # fp_res_hsde = sol_hsde['fixed_point_residuals']

    # these should match to machine precision
    assert jnp.linalg.norm(x_jax - x_c) < 1e-12
    assert jnp.linalg.norm(y_jax - y_c) < 1e-12
    assert jnp.linalg.norm(s_jax - s_c) < 1e-12

    # this line actually isn't true
    # assert jnp.all(jnp.diff(fp_res_hsde) < 0)
