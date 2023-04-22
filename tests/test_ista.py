import time
import jax.numpy as jnp
import scs
import numpy as np
from scipy.sparse import csc_matrix
from l2ws.algo_steps import k_steps_eval, k_steps_train_ista, create_projection_fn, lin_sys_solve
import jax.scipy as jsp
import pytest
import matplotlib.pyplot as plt
from l2ws.ista_model import ISTAmodel


@pytest.mark.skip(reason="temp")
def test_basic_ista():
    m, n = 5, 10
    A = jnp.array(np.random.normal(size=(m, n)))
    b = jnp.array(np.random.normal(size=(m)))
    k = 100
    z0 = jnp.zeros(n)
    lambd = .1
    evals,evecs = jnp.linalg.eigh(A.T@A)
    step = 1 / jnp.max(evals)

    z_k, losses = k_steps_train_ista(k, z0, b, lambd, A, step, supervised=False, z_star=None, jit=True)
    assert losses[-1] < losses[0] * .01 and losses[0] > .1


def test_train_ista():
    # ista setup
    N_train = 100
    N_test = 20
    N = N_train + N_test
    m, n = 5, 10
    A = jnp.array(np.random.normal(size=(m, n)))
    b_mat = jnp.array(np.random.normal(size=(N, m)))
    k = 100
    z0 = jnp.zeros(n)
    lambd = .1
    evals, evecs = jnp.linalg.eigh(A.T @ A)
    ista_step = 1 / jnp.max(evals)

    # setup inputs
    b_mat_train = b_mat[:N_train, :]
    b_mat_test = b_mat[N_train:, :]
    train_inputs = b_mat_train
    test_inputs = b_mat_test

    # create l2ws_model
    input_dict = dict(algorithm='ista',
                      train_unrolls=20, 
                      jit=False,
                      train_inputs=train_inputs, 
                      test_inputs=test_inputs,
                      b_mat_train=b_mat_train, 
                      b_mat_test=b_mat_test,
                      lambd=lambd,
                      ista_step=ista_step,
                      A=A
                      )
    l2ws_model = ISTAmodel(input_dict)

    # evaluate test before training
    init_test_loss, init_time_per_iter = l2ws_model.short_test_eval()

    # call train_batch without jitting
    params, state = l2ws_model.params, l2ws_model.state
    num_epochs = 100
    losses = jnp.zeros(num_epochs)
    for i in range(num_epochs):
        train_result = l2ws_model.train_full_batch(params, state)
        loss, params, state = train_result
        losses = losses.at[i].set(loss)

    l2ws_model.params, l2ws_model.state = params, state

    # evaluate test after training
    final_test_loss, final_time_per_iter = l2ws_model.short_test_eval()

    # import pdb
    # pdb.set_trace()