import time
import jax.numpy as jnp
import scs
import numpy as np
from scipy.sparse import csc_matrix
from l2ws.algo_steps import k_steps_eval_osqp, k_steps_train_osqp, create_projection_fn, lin_sys_solve, k_steps_train_osqp
import jax.scipy as jsp
import pytest
import matplotlib.pyplot as plt
from l2ws.ista_model import ISTAmodel
import cvxpy as cp
from jax import vmap
from functools import partial
from examples.ista import generate_b_mat
from scipy.spatial import distance_matrix
from examples.ista import sol_2_obj_diff, solve_many_probs_cvxpy
from l2ws.utils.nn_utils import get_nearest_neighbors


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
    z_k, losses, z_all = k_steps_eval_osqp(k, xy0, factor, A, q, rho, sigma, supervised=False, z_star=None, jit=True)
    assert losses[-1] < losses[0] * 1e-6 and losses[0] > .1

    # solve with cvxpy
    x = cp.Variable(n)
    z = cp.Variable(m)
    constraints = [l <= z, z <= u, A @ x == z]
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, P) + c @ x), constraints)
    prob.solve()

    assert jnp.linalg.norm(z_k[n + m:] - z.value) <= 1e-5
    assert jnp.linalg.norm(z_k[:n] - x.value) <= 1e-5


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
    z_k, losses, z_all = k_steps_eval_osqp(k, xy0, factor, A, q, rho, sigma, supervised=False, z_star=None, jit=True)
    import pdb
    pdb.set_trace()
    assert losses[-1] < losses[0] * 1e-6 and losses[0] > .1

    # # solve with cvxpy
    # x = cp.Variable(n)
    # z = cp.Variable(m)
    # constraints = [l <= z, z <= u, A @ x == z]
    # prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, P) + c @ x), constraints)
    # prob.solve()

    # assert jnp.linalg.norm(z_k[n + m:] - z.value) <= 1e-5
    # assert jnp.linalg.norm(z_k[:n] - x.value) <= 1e-5
