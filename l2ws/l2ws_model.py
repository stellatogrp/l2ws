import copy
import functools
from jax import jit, vmap
import jax.numpy as jnp
import jax
from jax import random
import optax
import time
from jaxopt import OptaxSolver
from utils.nn_utils import init_network_params, \
    predict_y
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pd


class L2WSmodel(object):
    def __init__(self, dict):
        proj, lin_sys_solve = dict['proj'], dict['lin_sys_solve']
        self.static_flag = dict['static_flag']
        self.batch_size = dict['nn_cfg'].batch_size
        self.epochs = dict['nn_cfg'].epochs
        self.lr = dict['nn_cfg'].lr
        self.decay_lr = dict['nn_cfg'].decay_lr
        self.min_lr = dict['nn_cfg'].min_lr

        self.eval_unrolls = dict['eval_unrolls']
        self.train_unrolls = dict['train_unrolls']

        self.train_inputs = dict['train_inputs']
        self.test_inputs = dict['test_inputs']

        self.N_train, _ = self.train_inputs.shape
        self.N_test, _ = self.test_inputs.shape
        self.batch_size = min([self.batch_size, self.N_train])
        self.num_batches = int(self.N_train/self.batch_size)

        self.y_stars_train = dict['y_stars_train']
        self.y_stars_test = dict['y_stars_test']
        self.x_stars_train = dict['x_stars_train']
        self.x_stars_test = dict['x_stars_test']
        self.w_stars_train = dict['w_stars_train']
        self.w_stars_test = dict['w_stars_test']

        self.q_mat_train = dict['q_mat_train']
        self.q_mat_test = dict['q_mat_test']

        self.m, self.n = dict['m'], dict['n']

        if self.static_flag:
            self.static_M = dict['static_M']
            self.static_algo_factor = dict['static_algo_factor']
        else:
            self.M_tensor_train = dict['M_tensor_train']
            self.M_tensor_test = dict['M_tensor_test']
            self.static_M, self.static_algo_factor = None, None
            self.matrix_invs_train = dict['matrix_invs_train']
            self.matrix_invs_test = dict['matrix_invs_test']

        self.nn_cfg = dict['nn_cfg']
        input_size = self.train_inputs.shape[1]
        self.prediction_variable = dict['prediction_variable']
        if self.prediction_variable == 'w':
            output_size = self.n + self.m
        elif self.prediction_variable == 'x':
            output_size = self.n
        layer_sizes = [input_size] + \
            self.nn_cfg['intermediate_layer_sizes'] + [output_size]
        self.params = init_network_params(layer_sizes, random.PRNGKey(0))
        self.state = None

        self.epoch = 0
        self.batched_predict_y = vmap(predict_y, in_axes=(None, 0))

        train_loss_dict = {'static_flag': self.static_flag,
                           'lin_sys_solve': lin_sys_solve,
                           'proj': proj,
                           'unrolls': self.train_unrolls,
                           'm': self.m,
                           'n': self.n,
                           'prediction_variable': self.prediction_variable,
                           'M_static': self.static_M,
                           'factor_static': self.static_algo_factor,
                           'diff_required': True
                           }
        eval_loss_dict = {'static_flag': self.static_flag,
                          'lin_sys_solve': lin_sys_solve,
                          'proj': proj,
                          'unrolls': self.eval_unrolls,
                          'm': self.m,
                          'n': self.n,
                          'prediction_variable': self.prediction_variable,
                          'M_static': self.static_M,
                          'factor_static': self.static_algo_factor,
                          'diff_required': False
                          }
        fixed_ws_dict = {'static_flag': self.static_flag,
                          'lin_sys_solve': lin_sys_solve,
                          'proj': proj,
                          'unrolls': self.eval_unrolls,
                          'm': self.m,
                          'n': self.n,
                          'prediction_variable': 'x',
                          'M_static': self.static_M,
                          'factor_static': self.static_algo_factor,
                          'diff_required': False
                          }
        self.loss_fn_train = create_loss_fn(train_loss_dict)
        self.loss_fn_eval = create_loss_fn(eval_loss_dict)

        # added fixed warm start eval
        self.loss_fn_fixed_ws = create_loss_fn(fixed_ws_dict)

        self.loss_fn_tests = {}

        if self.nn_cfg.method == 'adam':
            self.optimizer = OptaxSolver(opt=optax.adam(
                self.lr), fun=self.loss_fn_train, has_aux=True)
        elif self.nn_cfg.method == 'sgd':
            self.optimizer = OptaxSolver(opt=optax.sgd(
                self.lr), fun=self.loss_fn_train, has_aux=True)

        self.input_dict = dict
        self.tr_losses = None
        self.te_losses = None
        self.saveable_model = copy.copy(dict)

        self.saveable_model['tr_losses'] = self.tr_losses
        self.saveable_model['te_losses'] = self.te_losses

        self.saveable_model['pi'] = None
        self.saveable_model['proxf'] = None

        self.train_data = []
        
        self.state = self.optimizer.init_state(self.params)
        self.tr_losses_batch = []
        self.te_losses = []
        

    def pretrain(self, num_iters, stepsize=.001, method='adam', df_pretrain=None):
        # create pretrain function
        def pretrain_loss(params, inputs, targets):
            y_dual = self.batched_predict_y(params, inputs)
            pretrain_loss = jnp.mean(jnp.sum((y_dual - targets)**2, axis=1))
            return pretrain_loss

        if method == 'adam':
            optimizer_pretrain = OptaxSolver(
                opt=optax.adam(stepsize), fun=pretrain_loss, jit=True)
        elif method == 'sgd':
            optimizer_pretrain = OptaxSolver(
                opt=optax.sgd(stepsize), fun=pretrain_loss, jit=True)
        state = optimizer_pretrain.init_state(self.params)
        params = self.params
        pretrain_losses = np.zeros(num_iters)
        pretrain_test_losses = np.zeros(num_iters)

        if self.prediction_variable == 'w':
            train_targets = self.w_stars_train
            test_targets = self.w_stars_test
        elif self.prediction_variable == 'x':
            train_targets = self.x_stars_train
            test_targets = self.x_stars_test

        for _ in range(num_iters):
            out = optimizer_pretrain.update(params=params,
                                            state=state,
                                            inputs=self.train_inputs,
                                            targets=train_targets)

            params, state = out

            pretrain_losses[_] = state.value

            pretrain_test_losses[_] = pretrain_loss(
                params, self.test_inputs, test_targets)
            if _ % 10 == 0:
                print(
                    f"[Step {state.iter_num}] train loss: {state.value:.6f}")
                print(
                    f"[Step {state.iter_num}] test loss: {pretrain_test_losses[_]:.6f}")
            if df_pretrain is not None:
                data = np.vstack([pretrain_losses, pretrain_test_losses])
                data = data.T
                df_pretrain = pd.DataFrame(
                    data, columns=['pretrain losses', 'pretrain_test_losses'])
                df_pretrain.to_csv('pretrain_results.csv')

        self.params = params
        self.state = state
        return pretrain_losses, pretrain_test_losses

    def train_batch(self, batch_indices, decay_lr_flag=False, writer=None, logf=None):
        batch_inputs = self.train_inputs[batch_indices, :]
        batch_q_data = self.q_mat_train[batch_indices, :]

        # check if we need to update lr
        if decay_lr_flag:
            if self.min_lr <= self.lr * self.decay_lr and self.decay_lr < 1.0:
                # re-initialize the optimizer
                self.lr = self.lr * self.decay_lr
                print(f"lr decayed to {self.lr}")
                self.optimizer = OptaxSolver(opt=optax.adam(
                    self.lr), fun=self.loss_fn_train, has_aux=True)
                self.state = self.optimizer.init_state(self.params)

        t0 = time.time()

        if self.static_flag:
            results = self.optimizer.update(params=self.params,
                                            state=self.state,
                                            inputs=batch_inputs,
                                            q=batch_q_data,
                                            iters=self.train_unrolls)
        else:
            batch_inv_data = self.matrix_invs_train[batch_indices, :, :]
            batch_M_data = self.M_tensor_train[batch_indices, :, :]
            results = self.optimizer.update(params=self.params,
                                            state=self.state,
                                            inputs=batch_inputs,
                                            factor=batch_inv_data,
                                            M=batch_M_data,
                                            q=batch_q_data,
                                            iters=self.train_unrolls)
        self.params, self.state = results

        t1 = time.time()
        time_per_batch = (t1 - t0)
        print('time per batch', time_per_batch)
        train_out = self.state.aux

        print(
            f"[Step {self.state.iter_num}] train loss: {self.state.value:.6f}")

        if self.static_flag:
            test_loss, test_out, time_per = self.evaluate(self.train_unrolls,
                                                          self.test_inputs,
                                                          self.q_mat_test)
        else:
            eval_out = self.dynamic_eval(self.train_unrolls,
                                         self.test_inputs,
                                         self.matrix_invs_test,
                                         self.M_tensor_test,
                                         self.q_mat_test)
            test_loss, test_out, time_per = eval_out

        row = np.array([self.state.value, test_loss])
        self.train_data.append(pd.Series(row))
        self.tr_losses_batch.append(self.state.value)
        self.te_losses.append(test_loss)
        last10 = np.array(self.tr_losses_batch[-10:])
        moving_avg_train = last10.mean()
        if writer is not None:
            writer.writerow({
                'iter': self.state.iter_num,
                'train_loss': self.state.value,
                'moving_avg_train': moving_avg_train,
                'test_loss': test_loss
            })
            logf.flush()

    def evaluate(self, k, inputs, q, tag='test', fixed_ws=False):
        if fixed_ws:
            curr_loss_fn = self.loss_fn_fixed_ws
        else:
            curr_loss_fn = self.loss_fn_eval
        num_probs, _ = inputs.shape
        test_time0 = time.time()
        loss, out = curr_loss_fn(
            self.params, inputs, q, k)
        time_per_prob = (time.time() - test_time0)/num_probs
        print('eval time per prob', time_per_prob)
        print(f"[Epoch {self.epoch}] [k {k}] {tag} loss: {loss:.6f}")
        return loss, out, time_per_prob

    def dynamic_eval(self, k, inputs, matrix_inv, M, q, tag='test', fixed_ws=False):
        if fixed_ws:
            curr_loss_fn = self.loss_fn_fixed_ws
        else:
            curr_loss_fn = self.loss_fn_eval
        num_probs, _ = inputs.shape
        test_time0 = time.time()
        loss, out = curr_loss_fn(
            self.params, inputs, matrix_inv, M, q, k)
        time_per_prob = (time.time() - test_time0)/num_probs
        print('eval time per prob', time_per_prob)
        print(f"[Epoch {self.epoch}] [k {k}] {tag} loss: {loss:.6f}")
        
        return loss, out, time_per_prob


    def save(self):
        '''
        save the model itself
        this will also save the losses
        '''

        path = self.work_dir

        self.saveable_model['params'] = self.params
        self.saveable_model['state'] = self.state

        pkl_filename = path + '.pkl'
        with open(pkl_filename, 'wb') as f:
            pkl.dump(self.saveable_model, f)

        if self.train_data is not None:
            columns_train = ['train_loss', 'train_obj',
                             'rel_train_subopt', 'train_infeas']
            columns_test = ['test_loss', 'test_obj',
                            'rel_test_subopt', 'test_infeas']

            df = pd.DataFrame(self.train_data, columns=columns_train)
            df.to_csv(self.work_dir + 'results_train_data.csv')
            df = pd.DataFrame(self.test_data, columns=columns_test)
            df.to_csv(self.work_dir + 'results_test_data.csv')



def create_loss_fn(input_dict):
    static_flag = input_dict['static_flag']
    lin_sys_solve, proj = input_dict['lin_sys_solve'], input_dict['proj']
    unrolls = input_dict['unrolls']
    m, n = input_dict['m'], input_dict['n']
    prediction_variable = input_dict['prediction_variable']
    diff_required = input_dict['diff_required']

    # if dynamic, the next 2 set to None
    M_static, factor_static = input_dict['M_static'], input_dict['factor_static']

    def fixed_point(z_init, factor, q):
        u_tilde = lin_sys_solve(factor, z_init - q)
        u_temp = (2*u_tilde - z_init)
        u = proj(u_temp)
        # u^k, v^k = (x^k, y^k), (0, s^k)
        v = u + z_init - 2*u_tilde
        z = z_init + u - u_tilde
        #
        return z, u, v

    def predict(params, input, q, iters, factor, M):
        P, A = M[:n, :n], -M[n:, :n]
        b, c = q[n:], q[:n]


        if prediction_variable == 'w':
            uu = predict_y(params, input)
            w0 = M@uu + uu + q
            # w0 = uu
            x_primal = w0
        elif prediction_variable == 'x':
            # w0, x_primal = get_w0(params, input, q)
            w0 = input
            x_primal = w0

        z = w0
        iter_losses = jnp.zeros(iters)
        primal_residuals = jnp.zeros(iters)
        dual_residuals = jnp.zeros(iters)
        all_x_primals = jnp.zeros((iters, n))

        if diff_required:
            def _fp(i, val):
                z, loss_vec = val
                z_next, u, v = fixed_point(z, factor, q)
                diff = jnp.linalg.norm(z_next - z)
                loss_vec = loss_vec.at[i].set(diff)
                return z_next, loss_vec
            val = z, iter_losses
            out = jax.lax.fori_loop(0, iters, _fp, val)
            z, iter_losses = out
        else:
            def _fp(i, val):
                MM = M
                z, loss_vec, primal_residuals, dual_residuals = val
                z_next, u, v = fixed_point(z, factor, q)
                diff = jnp.linalg.norm(z_next - z)
                loss_vec = loss_vec.at[i].set(diff)

                pr = jnp.linalg.norm(A @ u[:n] + v[n:] - b)
                dr = jnp.linalg.norm(A.T @ u[n:] + P @ u[:n] + c)
                primal_residuals = primal_residuals.at[i].set(pr)
                dual_residuals = dual_residuals.at[i].set(dr)
                return z_next, loss_vec, primal_residuals, dual_residuals
            val = z, iter_losses, primal_residuals, dual_residuals
            out = jax.lax.fori_loop(0, iters, _fp, val)
            z, iter_losses, primal_residuals, dual_residuals = out

        # unroll 1 more time for the loss
        u_tilde = lin_sys_solve(factor, z - q)
        u_temp = 2*u_tilde - z
        u = proj(u_temp)
        z_next = z + u - u_tilde
        loss = jnp.linalg.norm(z_next - z)
        out = x_primal, z_next, u, all_x_primals

        if diff_required:
            return loss, iter_losses, out
        else:
            return loss, iter_losses, primal_residuals, dual_residuals, out

    if diff_required:
        out_axes = (0, 0, (0, 0, 0, 0))
    else:
        out_axes = (0, 0, 0, 0, (0, 0, 0, 0))

    if static_flag:
        predict_final = functools.partial(predict,
                                          factor=factor_static,
                                          M=M_static
                                          )
        batch_predict = vmap(predict_final, in_axes=(
            None, 0, 0, None), out_axes=out_axes)

        # @functools.partial(jax.jit, static_argnums=(3,))
        def loss_fn(params, inputs, q, iters):
            if diff_required:
                losses, iter_losses, out = batch_predict(
                    params, inputs, q, iters)
                loss_out = out, losses, iter_losses

            else:
                losses, iter_losses, primal_residuals, dual_residuals, out = batch_predict(
                    params, inputs, q, iters)
                loss_out = out, losses, iter_losses, primal_residuals, dual_residuals

            return losses.mean(), loss_out
    else:
        batch_predict = vmap(predict, in_axes=(
            None, 0, 0, None, 0, 0), out_axes=out_axes)

        @functools.partial(jax.jit, static_argnums=(5,))
        def loss_fn(params, inputs, factor, M, q, iters):
            if diff_required:
                losses, iter_losses, out = batch_predict(
                    params, inputs, q, iters, factor, M)
                loss_out = out, losses, iter_losses
            else:
                losses, iter_losses, primal_residuals, dual_residuals, out = batch_predict(
                    params, inputs, q, iters, factor, M)
                loss_out = out, losses, iter_losses, primal_residuals, dual_residuals
            return losses.mean(), loss_out
    return loss_fn
