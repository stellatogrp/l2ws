import time
import jax.numpy as jnp
import scs
import numpy as np
from scipy.sparse import csc_matrix
from l2ws.algo_steps import k_steps_eval, k_steps_train_ista, create_projection_fn, lin_sys_solve
import jax.scipy as jsp
import pytest
import matplotlib.pyplot as plt

def test_basic_ista():
    m, n = 5, 10
    A = jnp.array(np.random.normal(size=(m, n)))
    b = jnp.array(np.random.normal(size=(m)))
    k = 100
    z0 = jnp.zeros(n)
    lambd = .1
    evals,evecs = jnp.linalg.eigh(A.T@A)
    t = .01

    z_k, losses = k_steps_train_ista(k, z0, b, lambd, A, t, supervised=False, z_star=None, jit=True)
    assert losses[-1] < losses[0] * .1

def test_train_ista():
    N = 100
    m, n = 5, 10
    A = jnp.array(np.random.normal(size=(m, n)))
    b_mat = jnp.array(np.random.normal(size=(N, m)))
    k = 100
    z0 = jnp.zeros(n)
    lambd = .1
    evals,evecs = jnp.linalg.eigh(A.T @ A)
    t = .01

    z_k, losses = k_steps_train_ista(k, z0, b, lambd, A, t, supervised=False, z_star=None, jit=True)
    assert losses[-1] < losses[0] * .1