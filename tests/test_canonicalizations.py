import cvxpy as cp
import jax.numpy as jnp
import numpy as np

from l2ws.examples.robust_ls import multiple_random_robust_ls
from l2ws.examples.sparse_pca import multiple_random_sparse_pca
from l2ws.scs_problem import scs_jax


def test_phase_retrieval():
    pass


def test_sparse_pca():
    n_orig, k, r, N = 60, 5, 10, 5

    # create n parametric problems
    P, A, cones, q_mat, theta_mat_jax, A_tensor = multiple_random_sparse_pca(
        n_orig, k, r, N, factor=False)

    # scs hyperparams
    rho_x = 1
    scale = 10
    alpha = 1

    # solve with our DR splitting
    m, n = A.shape
    x_ws = jnp.ones(n)
    y_ws = jnp.ones(m)
    s_ws = jnp.zeros(m)
    max_iters = 800
    c, b = q_mat[0, :n], q_mat[0, n:]
    data = dict(P=P, A=A, c=c, b=b, cones=cones, x=x_ws, y=y_ws, s=s_ws)
    sol_hsde = scs_jax(data, hsde=True, rho_x=rho_x, scale=scale, alpha=alpha, iters=max_iters, plot=False)
    x_jax = sol_hsde['x']
    fp_res_hsde = sol_hsde['fixed_point_residuals']

    # form matrix from vector solution
    jax_obj = c @ x_jax

    # solve with cvxpy
    X = cp.Variable((n_orig, n_orig), symmetric=True)
    constraints = [X >> 0, cp.sum(cp.abs(X)) <= k, cp.trace(X) == 1]
    prob = cp.Problem(cp.Minimize(-cp.trace(A_tensor[0, :, :] @ X)), constraints)
    # prob.solve(solver=cp.SCS, verbose=True, rho_x=1, normalize=False, adaptive_scale=False)
    # prob.solve(solver=cp.SCS, verbose=True, rho_x=1, normalize=False)
    prob.solve(solver=cp.SCS, verbose=True, rho_x=rho_x, scale=scale,
               alpha=alpha, eps_abs=1e-4, eps_rel=0, adaptive_scale=False)
    cvxpy_obj = prob.value

    assert jnp.abs((jax_obj - cvxpy_obj) / cvxpy_obj) <= 1e-2

    assert fp_res_hsde[0] > 10
    assert fp_res_hsde[-1] < 5e-2 and fp_res_hsde[-1] > 1e-16


def test_robust_ls():
    m_orig, n_orig = 10, 20
    N = 1
    rho, b_center, b_range = 2, 1, 1

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
    x_jax = sol_hsde['x']
    fp_res_hsde = sol_hsde['fixed_point_residuals']
    x_jax_final = x_jax[:n_orig]

    # solve with cvxpy
    x, u, v = cp.Variable(n_orig), cp.Variable(), cp.Variable()
    constraints = [x >= 0, cp.norm(A_orig @ x - b_orig_mat[0, :]) <= u, cp.norm(x) <= v]
    prob = cp.Problem(cp.Minimize(u + rho * v), constraints)
    prob.solve()
    x_cvxpy = x.value

    assert jnp.linalg.norm(x_cvxpy - x_jax_final) <= 5e-2
    # assert jnp.all(jnp.diff(fp_res_hsde[1:]) < 1e-10)

    assert fp_res_hsde[0] > 10
    assert fp_res_hsde[-1] < 1e-3 and fp_res_hsde[-1] > 1e-16
