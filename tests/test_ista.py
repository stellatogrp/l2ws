import time
import jax.numpy as jnp
import scs
import numpy as np
from scipy.sparse import csc_matrix
from l2ws.algo_steps import k_steps_eval, k_steps_train_ista, create_projection_fn, lin_sys_solve, k_steps_train_fista
import jax.scipy as jsp
import pytest
import matplotlib.pyplot as plt
from l2ws.ista_model import ISTAmodel
import cvxpy as cp
from jax import vmap
from functools import partial
from examples.lasso import generate_b_mat
from scipy.spatial import distance_matrix
from examples.lasso import sol_2_obj_diff, solve_many_probs_cvxpy
from l2ws.utils.nn_utils import get_nearest_neighbors


# @pytest.mark.skip(reason="temp")
def test_basic_ista():
    m, n = 5, 10
    A = jnp.array(np.random.normal(size=(m, n)))
    b = jnp.array(np.random.normal(size=(m)))
    k = 100
    z0 = jnp.zeros(n)
    lambd = .1
    evals,evecs = jnp.linalg.eigh(A.T @ A)
    step = 1 / jnp.max(evals)

    z_k, losses = k_steps_train_ista(k, z0, b, lambd, A, step, supervised=False, z_star=None, jit=True)
    assert losses[-1] < losses[0] * .05 and losses[0] > .05


# @pytest.mark.skip(reason="temp")
def test_basic_fista():
    m, n = 5, 10
    A = jnp.array(np.random.normal(size=(m, n)))
    b = jnp.array(np.random.normal(size=(m)))
    k = 1000
    z0 = jnp.zeros(n)
    lambd = .1
    evals, mevecs = jnp.linalg.eigh(A.T@A)
    step = 1 / jnp.max(evals)

    z_k, losses = k_steps_train_fista(k, z0, b, lambd, A, step, supervised=False, z_star=None, jit=True)
    assert losses[-1] < losses[0] * 1e-5 and losses[0] > .01


# @pytest.mark.skip(reason="temp")
def test_train_ista():
    # ista setup
    N_train = 100
    N_test = 20
    N = N_train + N_test
    # m, n = 50, 100
    m, n = 100, 50
    A = jnp.array(np.random.normal(size=(m, n)))
    b_mat = jnp.array(np.random.normal(size=(N, m)))
    # b_mat = generate_b_mat(A, N)
    k = 1000
    z0 = jnp.zeros(n)
    lambd = .005
    evals, evecs = jnp.linalg.eigh(A.T @ A)
    ista_step = 1 / jnp.max(evals)

    # setup inputs
    b_mat_train = b_mat[:N_train, :]
    b_mat_test = b_mat[N_train:, :]
    train_inputs = b_mat_train
    test_inputs = b_mat_test

    # solve with cvxpy
    z_stars, objvals = solve_many_probs_cvxpy(A, b_mat, lambd)
    objvals_train = objvals[:N_train]
    objvals_test = objvals[N_train:]
    z_stars_train = z_stars[:N_train, :]
    z_stars_test = z_stars[N_train:, :]

    # create relative obj fn
    partial_rel = partial(sol_2_obj_diff, A=A, lambd=lambd)
    batch_rel = vmap(partial_rel, in_axes=(0, 0, 0), out_axes=(0))
    batch_rel_mat = vmap(batch_rel, in_axes=(1, None, None), out_axes=(0))

    # create l2ws_model
    # nn_cfg = {'lr': 1e-5} 
    nn_cfg = {}
    # nn_cfg = {'intermediate_layer_sizes': [200]} #, 'lr': 1e-1}
    train_unrolls = 50
    input_dict = dict(#algorithm='ista',
                      supervised=False,
                      train_unrolls=train_unrolls, 
                      jit=True,
                      train_inputs=train_inputs, 
                      test_inputs=test_inputs,
                      b_mat_train=b_mat_train, 
                      b_mat_test=b_mat_test,
                      lambd=lambd,
                      ista_step=ista_step,
                      A=A,
                      nn_cfg=nn_cfg,
                      z_stars_train=z_stars_train,
                      z_stars_test=z_stars_test,
                      )
    l2ws_model = ISTAmodel(input_dict)

    # full evaluation on the test set with nearest neighbor
    nearest_neighbors_z = get_nearest_neighbors(train_inputs, test_inputs, z_stars_train)
    nn_eval_out = l2ws_model.evaluate(k, nearest_neighbors_z, b_mat_test, z_stars=z_stars_test, fixed_ws=True, tag='test')
    nn_z_all = nn_eval_out[1][3]
    nn_rel_objs = batch_rel_mat(nn_z_all, b_mat_test, objvals_test).mean(axis=1)
    nn_losses = nn_eval_out[1][1].mean(axis=0)

    # evaluate test before training
    init_test_loss, init_time_per_iter = l2ws_model.short_test_eval()

    # full evaluation on the test set
    init_eval_out = l2ws_model.evaluate(k, test_inputs, b_mat_test, z_stars=z_stars_test, fixed_ws=False, tag='test')
    init_test_losses = init_eval_out[1][1].mean(axis=0)
    init_z_all = init_eval_out[1][3]
    init_rel_objs = batch_rel_mat(init_z_all, b_mat_test, objvals_test).mean(axis=1)

    # full evaluation on the train set
    # init_train_out = l2ws_model.evaluate(k, train_inputs, b_mat_train, z_stars=z_stars_train, fixed_ws=False, tag='test')
    # init_train_losses = init_train_out[1][1].mean(axis=0)
    # init_train_z_all = init_eval_out[1][3]
    # init_rel_objs = batch_rel_mat(init_train_z_all, b_mat_train, objvals_train).mean(axis=1)

    # call train_batch without jitting
    params, state = l2ws_model.params, l2ws_model.state
    num_epochs = 1000
    losses = jnp.zeros(num_epochs)
    for i in range(num_epochs):
        train_result = l2ws_model.train_full_batch(params, state)
        loss, params, state = train_result
        losses = losses.at[i].set(loss)

    l2ws_model.params, l2ws_model.state = params, state

    # evaluate test after training
    final_test_loss, final_time_per_iter = l2ws_model.short_test_eval()

    # full evaluation on the test set
    final_eval_out = l2ws_model.evaluate(k, test_inputs, b_mat_test, z_stars=z_stars_test, fixed_ws=False, tag='test')
    final_z_all = final_eval_out[1][3]
    final_rel_objs = batch_rel_mat(final_z_all, b_mat_test, objvals_test).mean(axis=1)

    final_test_losses = final_eval_out[1][1].mean(axis=0)

    assert final_test_loss < init_test_loss * .1
    # plt.plot(init_test_losses, label='cold start')
    # plt.plot(final_test_losses, label=f"learned k={train_unrolls}")
    # plt.plot(nn_losses, label='nearest neighbor')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.title('fixed point residuals')
    # plt.legend()
    # plt.show()

    # plt.plot(init_rel_objs, label='cold start')
    # plt.plot(final_rel_objs, label=f"learned k={train_unrolls}")
    # plt.plot(nn_rel_objs, label='nearest neighbor')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.title('objective suboptimality')
    # plt.legend()
    # plt.show()

    # plt.plot(losses)
    # plt.yscale('log')
    # plt.show()

    import pdb
    pdb.set_trace()
