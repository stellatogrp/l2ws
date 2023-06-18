import time
import jax.numpy as jnp
import scs
import numpy as np
from scipy.sparse import csc_matrix
from l2ws.algo_steps import k_steps_train_extragrad
import jax.scipy as jsp
import pytest
import matplotlib.pyplot as plt
from l2ws.eg_model import EGmodel
import cvxpy as cp
from jax import vmap
from functools import partial
from examples.lasso import generate_b_mat
from scipy.spatial import distance_matrix
from examples.lasso import sol_2_obj_diff, solve_many_probs_cvxpy
from l2ws.utils.nn_utils import get_nearest_neighbors
from jaxopt import OptaxSolver
import optax


@pytest.mark.skip(reason="temp")
def test_basic_extragrad():
    m, n = 5, 10
    Q = .1 * jnp.eye(n)
    R = .1 * jnp.eye(m)
    A = jnp.array(np.random.normal(size=(m, n)))
    c = jnp.array(np.random.normal(size=(n)))
    b = jnp.array(np.random.normal(size=(m)))
    k = 500
    z0 = jnp.zeros(m + n)
    # evals, evecs = jnp.linalg.eigh(A.T @ A)
    # step = 1 / jnp.max(evals)
    step = .1

    # stack into q
    q = jnp.zeros(m * n + m + n)
    q = q.at[:m * n].set(jnp.ravel(A))
    q = q.at[m * n: m * n + n].set(c)
    q = q.at[m * n + n:].set(b)

    z_k, losses = k_steps_train_extragrad(k, z0, q, Q, R, step, supervised=False, z_star=None, jit=True)

    plt.plot(losses)
    plt.yscale('log')
    plt.show()
    assert losses[-1] < losses[0] * .01 and losses[0] > .05

    import pdb
    pdb.set_trace()




# @pytest.mark.skip(reason="temp")
def test_train_extragrad():
    # ista setup
    N_train = 5000
    N_test = 20
    N = N_train + N_test
    m, n = 20, 20

    Q = 1 * jnp.eye(n)
    R = 1 * jnp.eye(m)
    A0 = jnp.array(np.random.normal(size=(m, n)))
    A_tensor = 0*A0 + 1 * jnp.array(np.random.normal(size=(N, m, n))) #jnp.ones((N, m, n)) #
    c_mat = jnp.array(np.random.normal(size=(N, n)))
    b_mat = jnp.array(np.random.normal(size=(N, m)))
    A_stacked = jnp.reshape(A_tensor, (N, m * n))

    q_mat = jnp.hstack([A_stacked, c_mat, b_mat])

    k = 50
    z0 = jnp.zeros(m + n)
    # lambd = .005
    # evals, evecs = jnp.linalg.eigh(A.T @ A)
    eg_step = .05

    # setup inputs
    q_mat_train = q_mat[:N_train, :]
    q_mat_test = q_mat[N_train:, :]
    train_inputs = q_mat_train
    test_inputs = q_mat_test

    # solve to optimality (needed for nearest neighbors)
    # k_steps_batch = vmap(k_steps_train_extragrad, in_axes=, out)
    z_stars = jnp.zeros((N, m + n))
    for i in range(N):
        z_k, losses = k_steps_train_extragrad(k, z0, q_mat[i, :], Q, R, eg_step, supervised=False, z_star=None, jit=True)
        print(i, losses[-1])
        # plt.plot(losses)
        # plt.yscale('log')
        # plt.show()
        # return
        z_stars = z_stars.at[i, :].set(z_k)
    z_stars_train = z_stars[:N_train, :]
    z_stars_test = z_stars[N_train:, :]

    # solve with cvxpy
    # z_stars, objvals = solve_many_probs_cvxpy(A, b_mat, lambd)
    # objvals_train = objvals[:N_train]
    # objvals_test = objvals[N_train:]
    # z_stars_train = z_stars[:N_train, :]
    # z_stars_test = z_stars[N_train:, :]

    # create relative obj fn
    # partial_rel = partial(sol_2_obj_diff, A=A, lambd=lambd)
    # batch_rel = vmap(partial_rel, in_axes=(0, 0, 0), out_axes=(0))
    # batch_rel_mat = vmap(batch_rel, in_axes=(1, None, None), out_axes=(0))

    # create l2ws_model
    # nn_cfg = {'lr': 1e-5} 
    # nn_cfg = {}
    batch_size = 10
    nn_cfg = {'intermediate_layer_sizes': [100, 100], 'batch_size': batch_size, 'method': 'adam', 'lr': 1e-4}
    train_unrolls = 1
    input_dict = dict(supervised=True,
                      train_unrolls=train_unrolls, 
                      jit=True,
                      train_inputs=train_inputs, 
                      test_inputs=test_inputs,
                      Q=Q,
                      R=R,
                      q_mat_train=q_mat_train, 
                      q_mat_test=q_mat_test,
                      eg_step=eg_step,
                      nn_cfg=nn_cfg,
                      z_stars_train=z_stars_train,
                      z_stars_test=z_stars_test,
                      )
    l2ws_model = EGmodel(input_dict)

    # full evaluation on the test set with nearest neighbor
    nearest_neighbors_z = get_nearest_neighbors(train_inputs, test_inputs, z_stars_train)
    nn_eval_out = l2ws_model.evaluate(k, nearest_neighbors_z, q_mat_test, z_stars=z_stars_test, fixed_ws=True, tag='test')
    nn_z_all = nn_eval_out[1][3]
    # nn_rel_objs = batch_rel_mat(nn_z_all, b_mat_test, objvals_test).mean(axis=1)
    nn_losses = nn_eval_out[1][1].mean(axis=0)

    # evaluate test before training
    init_test_loss, init_time_per_iter = l2ws_model.short_test_eval()

    # full evaluation on the test set
    init_eval_out = l2ws_model.evaluate(k, test_inputs, q_mat_test, z_stars=z_stars_test, fixed_ws=False, tag='test')
    init_test_losses = init_eval_out[1][1].mean(axis=0)
    init_z_all = init_eval_out[1][3]
    # init_rel_objs = batch_rel_mat(init_z_all, b_mat_test, objvals_test).mean(axis=1)

    # full evaluation on the train set
    # init_train_out = l2ws_model.evaluate(k, train_inputs, b_mat_train, z_stars=z_stars_train, fixed_ws=False, tag='test')
    # init_train_losses = init_train_out[1][1].mean(axis=0)
    # init_train_z_all = init_eval_out[1][3]
    # init_rel_objs = batch_rel_mat(init_train_z_all, b_mat_train, objvals_train).mean(axis=1)

    # call train_batch without jitting
    params, state = l2ws_model.params, l2ws_model.state
    num_epochs = 1000
    # losses = jnp.zeros(num_epochs)
    losses = []
    num_batches = int(N_train / batch_size)
    for i in range(num_epochs):
        train_result = l2ws_model.train_full_batch(params, state)
        loss, params, state = train_result
        # losses = losses.at[i].set(loss)
        losses.append(loss)
        print(i, loss)
        # for j in range(num_batches):
        #     start = j * batch_size
        #     stop = (j + 1) * batch_size
        #     batch_indices = jnp.arange(start, stop)
        #     train_result = l2ws_model.train_batch(batch_indices, params, state)
        #     loss, params, state = train_result
        #     losses.append(loss)
        # print(i, np.array(losses[-num_batches:]).mean())
        
        # if i % 100 == 0:
        #     l2ws_model.supervised = False
        #     # if l2ws_model.train_unrolls == 20:
        #     #     l2ws_model.train_unrolls = 15
        #     # elif l2ws_model.train_unrolls == 15:
        #     #     l2ws_model.train_unrolls = 20
        #     l2ws_model.train_unrolls = 15
            
        #     new_dict = {'supervised': False}
        #     l2ws_model.create_all_loss_fns(new_dict)
        #     l2ws_model.optimizer = OptaxSolver(opt=optax.adam(
        #             l2ws_model.lr), fun=l2ws_model.loss_fn_train, has_aux=False)
        #     l2ws_model.state = l2ws_model.optimizer.init_state(l2ws_model.params)

    l2ws_model.params, l2ws_model.state = params, state

    # evaluate test after training
    final_test_loss, final_time_per_iter = l2ws_model.short_test_eval()

    # full evaluation on the test set
    final_eval_out = l2ws_model.evaluate(k, test_inputs, q_mat_test, z_stars=z_stars_test, fixed_ws=False, tag='test')
    final_z_all = final_eval_out[1][3]
    # final_rel_objs = batch_rel_mat(final_z_all, b_mat_test, objvals_test).mean(axis=1)

    final_test_losses = final_eval_out[1][1].mean(axis=0)

    plt.plot(losses)
    plt.yscale('log')
    plt.show()

    # assert final_test_loss < init_test_loss * .1
    plt.plot(init_test_losses, label='cold start')
    plt.plot(final_test_losses, label=f"learned k={train_unrolls}")
    plt.plot(nn_losses, label='nearest neighbor')
    plt.yscale('log')
    # plt.xscale('log')
    plt.title('fixed point residuals')
    plt.legend()
    plt.show()

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
