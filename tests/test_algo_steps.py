import time
from examples.robust_ls import random_robust_ls
import pytest
from l2ws.algo_steps import fixed_point, fixed_point_hsde, create_M, create_projection_fn, \
    lin_sys_solve
import jax.numpy as jnp
from l2ws.scs_problem import scs_jax
import pdb
import scs
import numpy as np
from scipy.sparse import csc_matrix


def test_nonexpansiveness():
    pass


# def test_jit_speed():
#     t0_non_jit = time.time()
#     t1_non_jit = time.time()
#     non_jit_time = t1_non_jit - t0_non_jit

#     t0_jit = time.time()
#     t1_jit = time.time()
#     jit_time = t1_jit - t0_jit

#     assert jit_time - non_jit_time > 0


# def test_hsde_socp():
#     """
#     tests to make sure hsde returns the same solution as the non-hsde
#     tests socp of different cone sizes also (there are 2 SOCs)
#     """
#     # get a random robust least squares problem
#     m_orig, n_orig = 50, 55
#     rho = 1
#     b_center, b_range = 1, 1
#     P, A, c, b, cones = random_robust_ls(m_orig, n_orig, rho, b_center, b_range)

#     data = dict(P=P, A=A, c=c, b=b, cones=cones)

#     sol_std = scs_jax(data, hsde=False, iters=200)
#     x_std, y_std, s_std = sol_std['x'], sol_std['y'], sol_std['s']
#     fp_res_std = sol_std['fp_residuals']

#     sol_hsde = scs_jax(data, hsde=True, iters=200)
#     x_hsde, y_hsde, s_hsde = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
#     fp_res_hsde = sol_hsde['fp_residuals']

#     assert jnp.linalg.norm(x_hsde - x_std) < 1e-3
#     assert jnp.linalg.norm(y_hsde - y_std) < 1e-3
#     assert jnp.linalg.norm(s_hsde - s_std) < 1e-3
#     assert jnp.all(jnp.diff(fp_res_std) < 0)
#     assert jnp.all(jnp.diff(fp_res_hsde) < 0)


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
    fp_res_hsde = sol_hsde['fp_residuals']
    print('fp_res_hsde', fp_res_hsde)

    assert jnp.linalg.norm(x_jax - x_c) < 1e-3
    assert jnp.linalg.norm(y_jax - y_c) < 1e-3
    assert jnp.linalg.norm(s_jax - s_c) < 1e-3
    assert jnp.all(jnp.diff(fp_res_hsde) < 0)