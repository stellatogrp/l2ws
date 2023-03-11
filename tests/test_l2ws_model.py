import time
from examples.robust_ls import random_robust_ls, multiple_random_robust_ls
import jax.numpy as jnp
from l2ws.scs_problem import scs_jax
import scs
import numpy as np
from scipy.sparse import csc_matrix
from l2ws.l2ws_model import L2WSmodel
from l2ws.algo_steps import create_projection_fn, create_M
import jax.scipy as jsp


def test_minimal_l2ws_model():
    """
    tests that we can initialize an L2WSmodel with the minimal amount of information needed
    """
    # get the problem
    N_train, N_test = 10, 5
    N = N_train + N_test

    m_orig, n_orig = 30, 40
    rho = 1
    b_center, b_range = 1, 1
    P, A, cones, q_mat, theta_mat = multiple_random_robust_ls(
        m_orig, n_orig, rho, b_center, b_range, N)
    m, n = A.shape

    proj = create_projection_fn(cones, n)
    q_mat_train, q_mat_test = q_mat[:N_train, :], q_mat[N_train:N, :]
    train_inputs, test_inputs = theta_mat[:N_train, :], theta_mat[N_train:N, :]
    static_M = create_M(P, A)
    static_algo_factor = jsp.linalg.lu_factor(static_M + jnp.eye(n + m))

    # enter into the L2WSmodel
    input_dict = dict(m=m, n=n, hsde=True, static_flag=True, proj=proj,
                      train_unrolls=20, q_mat_train=q_mat_train, q_mat_test=q_mat_test,
                      train_inputs=train_inputs, test_inputs=test_inputs,
                      static_M=static_M, static_algo_factor=static_algo_factor)
    l2ws_model = L2WSmodel(input_dict)

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

    # final loss should be at least 50% better than the first loss
    assert losses[-1] / losses[0] < 0.5

    # evaluate test after training
    final_test_loss, final_time_per_iter = l2ws_model.short_test_eval()

    # test loss does not get 5% worse
    assert final_test_loss < init_test_loss * 1.05

    # after jitting, evaluating the test set should be much faster
    assert final_time_per_iter < .1 * init_time_per_iter
