import time
from examples.robust_ls import random_robust_ls, multiple_random_robust_ls
import jax.numpy as jnp
from l2ws.scs_problem import scs_jax
import scs
import numpy as np
from l2ws.algo_steps import create_projection_fn, create_M
import cvxpy as cp


def test_robust_ls():
    m_orig, n_orig = 10, 20
    N = 1
    rho, b_center, b_range = 1, 1, 1

    P, A, cones, q_mat, theta_mat, A_orig, b_orig_mat = multiple_random_robust_ls(
        m_orig, n_orig, rho, b_center, b_range, N)
    m, n = A.shape

    # solve with our DR splitting
    x_ws = np.ones(n)
    y_ws = np.ones(m)
    s_ws = np.zeros(m)
    max_iters = 500
    c, b = q_mat[0, :n], q_mat[0, n:]
    data = dict(P=P, A=A, c=c, b=b, cones=cones, x=x_ws, y=y_ws, s=s_ws)
    sol_hsde = scs_jax(data, hsde=True, iters=max_iters)
    x_jax, y_jax, s_jax = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    fp_res_hsde = sol_hsde['fixed_point_residuals']
    x_jax_final = x_jax[:n_orig]

    # solve with cvxpy
    x, u, v = cp.Variable(n_orig), cp.Variable(), cp.Variable()
    constraints = [x >= 0, cp.norm(A_orig @ x - b_orig_mat[0, :]) <= u, cp.norm(x) <= v]
    prob = cp.Problem(cp.Minimize(u + rho * v), constraints)
    prob.solve()
    x_cvxpy = x.value

    assert jnp.linalg.norm(x_cvxpy - x_jax_final) <= 1e-3
