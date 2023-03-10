from jax import jit, vmap
import jax.numpy as jnp
import jax
from jax import random
import optax
import time
from jaxopt import OptaxSolver
from l2ws.utils.nn_utils import init_network_params, \
    predict_y, init_matrix_params
from l2ws.utils.generic_utils import unvec_symm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from jax.config import config
from scipy.spatial import distance_matrix
import logging
from l2ws.algo_steps import lin_sys_solve, fixed_point, fixed_point_hsde
from functools import partial
config.update("jax_enable_x64", True)


class L2WSmodel(object):
    def __init__(self, dict):
        self.proj = dict['proj']
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
        self.w_stars_train = jnp.array(dict['w_stars_train'])
        self.w_stars_test = jnp.array(dict['w_stars_test'])
        self.u_stars_train = jnp.hstack([self.x_stars_train, self.y_stars_train])
        self.u_stars_test = jnp.hstack([self.x_stars_test, self.y_stars_test])

        self.q_mat_train = dict['q_mat_train']
        self.q_mat_test = dict['q_mat_test']

        self.angle_anchors = dict['angle_anchors']
        self.supervised = dict['supervised']

        self.m, self.n = dict['m'], dict['n']
        self.psd, self.tx, self.ty = dict.get('psd'), dict.get('tx', 0), dict.get('ty', 0)
        self.dx, self.dy = dict.get('dx', 0), dict.get('dy', 0)
        self.psd_size = dict.get('psd_size')
        self.low_2_high_dim = dict.get('low_2_high_dim')

        self.num_clusters = dict.get('num_clusters')
        self.x_psd_indices = dict.get('x_psd_indices')
        self.y_psd_indices = dict.get('y_psd_indices')
        self.loss_method = dict.get('loss_method')
        self.share_all = dict.get('share_all')
        self.pretrain_alpha = dict.get('pretrain_alpha')
        self.normalize_alpha = dict.get('normalize_alpha')
        self.plateau_decay = dict.get('plateau_decay')
        self.dont_decay_until = 2 * self.plateau_decay.avg_window_size
        self.epoch_decay_points = []

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

        self.batched_predict_y = vmap(predict_y, in_axes=(None, 0))

        if self.share_all:
            output_size = self.num_clusters
        else:
            if self.psd:
                n_x_non_psd = self.n - int(self.psd_size * (self.psd_size + 1) / 2)
                n_y_non_psd = self.m - int(self.psd_size * (self.psd_size + 1) / 2)
                n_x_low = n_x_non_psd + self.dx * self.psd_size
                n_y_low = n_y_non_psd + self.dy * self.psd_size

                output_size = n_x_low + n_y_low + self.tx + self.ty
            else:
                output_size = self.n + self.m

        layer_sizes = [input_size] + \
            self.nn_cfg['intermediate_layer_sizes'] + [output_size]

        self.nn_params = init_network_params(layer_sizes, random.PRNGKey(0))
        key = 0
        if self.share_all:
            out = self.cluster_z()
            self.Z_shared = out[0]
            self.train_cluster_indices, self.test_cluster_indices = out[1], out[2]
            self.X_list, self.Y_list = [], []
            self.params = self.nn_params
            if self.pretrain_alpha:
                self.pretrain_alphas(1000, None, share_all=True)

        else:
            self.Z_shared = None
            if self.psd and self.tx + self.ty > 0:
                if self.learn_XY:
                    self.X_list = init_matrix_params(self.tx, self.psd_size, random.PRNGKey(key))
                    key += self.tx
                    self.Y_list = init_matrix_params(self.ty, self.psd_size, random.PRNGKey(key))
                    self.params = self.nn_params + self.X_list + self.Y_list
                else:
                    out = self.cluster_init_XY_list()
                    self.X_list, self.Y_list = out[0], out[1]
                    self.train_cluster_indices, self.test_cluster_indices = out[2], out[3]
                    self.params = self.nn_params
                    if self.pretrain_alpha:
                        self.pretrain_alphas(1000, n_x_low + n_y_low)
            else:
                self.X_list, self.Y_list = [], []
                self.params = self.nn_params

        self.epoch = 0

        train_loss_dict = {'diff_required': True,
                           'unrolls': self.train_unrolls,
                           'bypass_nn': False,
                           'supervised': self.supervised
                           }
        eval_loss_dict = {'diff_required': False,
                          'unrolls': self.eval_unrolls,
                          'bypass_nn': False,
                          'supervised': False
                          }
        fixed_ws_dict = {'diff_required': False,
                         'unrolls': self.eval_unrolls,
                         'bypass_nn': True,
                         'supervised': False}

        # loss fn for training
        self.loss_fn_train = self.create_loss_fn(train_loss_dict)

        # loss fn for evaluation
        self.loss_fn_eval = self.create_loss_fn(eval_loss_dict)

        # added fixed warm start eval - bypasses neural network
        self.loss_fn_fixed_ws = self.create_loss_fn(fixed_ws_dict)

        if self.nn_cfg.method == 'adam':
            self.optimizer = OptaxSolver(opt=optax.adam(
                self.lr), fun=self.loss_fn_train, has_aux=False)
        elif self.nn_cfg.method == 'sgd':
            self.optimizer = OptaxSolver(opt=optax.sgd(
                self.lr), fun=self.loss_fn_train, has_aux=False)

        # self.input_dict = dict
        self.tr_losses = None
        self.te_losses = None

        self.train_data = []

        self.state = self.optimizer.init_state(self.params)
        self.tr_losses_batch = []
        self.te_losses = []

    def cluster_z(self):
        N_train = self.x_stars_train.shape[0]
        sample_indices = np.random.choice(N_train, self.num_clusters, replace=False)
        Z_shared = self.w_stars_train[sample_indices, :].T

        # compute distance matrix
        def get_indices(input, flag):
            distances = distance_matrix(np.array(input), np.array(Z_shared.T))
            print('distances psd', distances)
            indices = np.argmin(distances, axis=1)
            print('indices psd', indices)
            best_val = np.min(distances, axis=1)
            print('best val', best_val)
            plt.plot(indices)
            plt.savefig(f"{flag}_indices_psd_plot.pdf", bbox_inches='tight')
            plt.clf()
            return indices
        train_cluster_indices = get_indices(self.w_stars_train, 'train')
        test_cluster_indices = get_indices(self.w_stars_test, 'test')

        return Z_shared, train_cluster_indices, test_cluster_indices

    def cluster_init_XY_list(self):
        # put into matrix form -- use vec_symm
        X_list, Y_list = [], []

        N_train = self.x_stars_train.shape[0]
        sample_indices = np.random.choice(N_train, self.num_clusters, replace=False)
        for i in range(self.num_clusters):
            index = sample_indices[i]
            x_psd = self.x_stars_train[index, self.x_psd_indices]
            y_psd = self.y_stars_train[index, self.y_psd_indices]
            X, Y = unvec_symm(x_psd, self.psd_size), unvec_symm(y_psd, self.psd_size)
            X_list.append(X)
            Y_list.append(Y)

        # do clustering -- for now just take first self.num_clusters
        # clusters = self.u_stars_train[:self.num_clusters, :]
        clusters = self.u_stars_train[sample_indices, :]

        # compute distance matrix
        def get_indices(input, flag):
            distances = distance_matrix(np.array(input), np.array(clusters))
            print('distances psd', distances)
            indices = np.argmin(distances, axis=1)
            print('indices psd', indices)
            best_val = np.min(distances, axis=1)
            print('best val', best_val)
            plt.plot(indices)
            plt.savefig(f"{flag}_indices_psd_plot.pdf", bbox_inches='tight')
            plt.clf()
            return indices
        train_cluster_indices = get_indices(self.u_stars_train, 'train')
        test_cluster_indices = get_indices(self.u_stars_test, 'test')
        return X_list, Y_list, train_cluster_indices, test_cluster_indices

    def pretrain_alphas(self, num_iters, n_xy_low, share_all=False, stepsize=.001, method='adam',
                        batches=10):
        def pretrain_loss(params, inputs, targets):
            nn_output = self.batched_predict_y(params, inputs)
            if share_all:
                alpha_hat = nn_output
            else:
                alpha_hat = nn_output[:, n_xy_low:]
            pretrain_loss = jnp.mean(jnp.sum((alpha_hat - targets)**2, axis=1))
            return pretrain_loss

        maxiters = int(num_iters / batches)

        if method == 'adam':
            optimizer_pretrain = OptaxSolver(
                opt=optax.adam(stepsize), fun=pretrain_loss, jit=False, maxiter=maxiters)
        elif method == 'sgd':
            optimizer_pretrain = OptaxSolver(
                opt=optax.sgd(stepsize), fun=pretrain_loss, jit=True, maxiter=maxiters)
        state = optimizer_pretrain.init_state(self.params)
        params = self.params
        pretrain_losses = np.zeros(batches + 1)
        pretrain_test_losses = np.zeros(batches + 1)

        # do a 1-hot encoding - assume given vector of indices
        if share_all:
            train_targets = jax.nn.one_hot(self.train_cluster_indices, self.num_clusters)
            test_targets = jax.nn.one_hot(self.test_cluster_indices, self.num_clusters)
        else:
            train_targets_x = jax.nn.one_hot(self.train_cluster_indices, self.tx)
            train_targets_y = jax.nn.one_hot(self.train_cluster_indices, self.ty)
            test_targets_x = jax.nn.one_hot(self.test_cluster_indices, self.tx)
            test_targets_y = jax.nn.one_hot(self.test_cluster_indices, self.ty)
            train_targets = jnp.hstack([train_targets_x, train_targets_y])
            test_targets = jnp.hstack([test_targets_x, test_targets_y])

        curr_pretrain_loss = pretrain_loss(
            params, self.train_inputs, train_targets)
        curr_pretrain_test_loss = pretrain_loss(
            params, self.test_inputs, test_targets)
        pretrain_losses[0] = curr_pretrain_loss
        pretrain_test_losses[0] = curr_pretrain_test_loss

        for i in range(batches):
            out = optimizer_pretrain.run(init_params=params,
                                         inputs=self.train_inputs,
                                         targets=train_targets)
            params = out.params
            state = out.state
            curr_pretrain_test_loss = pretrain_loss(
                params, self.test_inputs, test_targets)
            pretrain_losses[i + 1] = state.value
            pretrain_test_losses[i + 1] = curr_pretrain_test_loss
            data = np.vstack([pretrain_losses, pretrain_test_losses])
            data = data.T
            df_pretrain = pd.DataFrame(
                data, columns=['pretrain losses', 'pretrain_test_losses'])
            df_pretrain.to_csv('pretrain_alpha_results.csv')

        self.params = params
        self.state = state
        return pretrain_losses, pretrain_test_losses

    def pretrain(self, num_iters, stepsize=.001, method='adam', df_pretrain=None, batches=1):
        # create pretrain function
        def pretrain_loss(params, inputs, targets):
            if self.tx is None or self.ty is None:
                nn_output = self.batched_predict_y(params, inputs)
                uu = nn_output
            else:
                num_nn_params = len(self.nn_params)

                def predict(params_, inputs_):
                    nn_params = params_[:num_nn_params]
                    nn_output = predict_y(nn_params, inputs_)
                    X_list = params_[num_nn_params:num_nn_params + self.tx]
                    Y_list = params_[num_nn_params + self.tx:]
                    uu = self.low_2_high_dim(nn_output, X_list, Y_list)
                    return uu
                batch_predict = vmap(predict, in_axes=(None, 0), out_axes=(0))
                uu = batch_predict(params, inputs)

            pretrain_loss = jnp.mean(jnp.sum((uu - targets)**2, axis=1))
            return pretrain_loss

        maxiters = int(num_iters / batches)

        if method == 'adam':
            optimizer_pretrain = OptaxSolver(
                opt=optax.adam(stepsize), fun=pretrain_loss, jit=True, maxiter=maxiters)
        elif method == 'sgd':
            optimizer_pretrain = OptaxSolver(
                opt=optax.sgd(stepsize), fun=pretrain_loss, jit=True, maxiter=maxiters)
        state = optimizer_pretrain.init_state(self.params)
        params = self.params
        pretrain_losses = np.zeros(batches + 1)
        pretrain_test_losses = np.zeros(batches + 1)

        train_targets = self.w_stars_train
        test_targets = self.w_stars_test

        curr_pretrain_loss = pretrain_loss(
            params, self.train_inputs, train_targets)
        curr_pretrain_test_loss = pretrain_loss(
            params, self.test_inputs, test_targets)
        pretrain_losses[0] = curr_pretrain_loss
        pretrain_test_losses[0] = curr_pretrain_test_loss

        for i in range(batches):
            out = optimizer_pretrain.run(init_params=params,
                                         inputs=self.train_inputs,
                                         targets=train_targets)
            params = out.params
            state = out.state
            curr_pretrain_test_loss = pretrain_loss(
                params, self.test_inputs, test_targets)
            pretrain_losses[i + 1] = state.value
            pretrain_test_losses[i + 1] = curr_pretrain_test_loss
            data = np.vstack([pretrain_losses, pretrain_test_losses])
            data = data.T
            df_pretrain = pd.DataFrame(
                data, columns=['pretrain losses', 'pretrain_test_losses'])
            df_pretrain.to_csv('pretrain_results.csv')

        self.params = params
        self.state = state
        return pretrain_losses, pretrain_test_losses

    def decay_upon_plateau(self):
        """
        this method decays the learning rate upon hitting a plateau
            on the training loss
        self.avg_window_plateau: take the last avg_window number of epochs and compared it
            against the previous avg_window number of epochs to compare
        self.plateau_decay_factor: multiplicative factor we decay the learning rate by
        self.plateau_tolerance: the tolerance condition decrease to check if we should decrease

        we decay the learn rate by decay_factor if
           self.tr_losses[-2*avg_window:-avg_window] - self.tr_losses[-avg_window:] <= tolerance
        """
        decay_factor = self.plateau_decay.decay_factor

        window_batches = self.plateau_decay.avg_window_size * self.num_batches
        plateau_tolerance = self.plateau_decay.tolerance
        patience = self.plateau_decay.patience

        if self.plateau_decay.min_lr <= self.lr / decay_factor:
            tr_losses_batch_np = np.array(self.tr_losses_batch)
            prev_window_losses = tr_losses_batch_np[-2*window_batches:-window_batches].mean()
            curr_window_losses = tr_losses_batch_np[-window_batches:].mean()
            print('prev_window_losses', prev_window_losses)
            print('curr_window_losses', curr_window_losses)
            plateau = prev_window_losses - curr_window_losses <= plateau_tolerance
            if plateau:
                # keep track of the learning rate
                self.lr = self.lr / decay_factor

                # update the optimizer (restart) and reset the state
                if self.nn_cfg.method == 'adam':
                    self.optimizer = OptaxSolver(opt=optax.adam(
                        self.lr), fun=self.loss_fn_train, has_aux=False)
                elif self.nn_cfg.method == 'sgd':
                    self.optimizer = OptaxSolver(opt=optax.sgd(
                        self.lr), fun=self.loss_fn_train, has_aux=False)
                self.state = self.optimizer.init_state(self.params)
                logging.info(f"the decay rate is now {self.lr}")

                # log the current decay epoch
                self.epoch_decay_points.append(self.epoch)

                # don't decay for another 2 * window number of epochs
                wait_time = 2 * patience * self.plateau_decay.avg_window_size
                self.dont_decay_until = self.epoch + wait_time

    def train_batch(self, batch_indices, params, state):
        batch_inputs = self.train_inputs[batch_indices, :]
        batch_q_data = self.q_mat_train[batch_indices, :]
        batch_z_stars = self.w_stars_train[batch_indices, :]
        results = self.optimizer.update(params=params,
                                        state=state,
                                        inputs=batch_inputs,
                                        q=batch_q_data,
                                        iters=self.train_unrolls,
                                        z_stars=batch_z_stars)
        params, state = results
        return state.value, params, state

    def short_test_eval(self):
        if self.static_flag:
            test_loss, test_out, time_per_prob = self.static_eval(self.train_unrolls,
                                                                  self.test_inputs,
                                                                  self.q_mat_test,
                                                                  self.w_stars_test)
        else:
            eval_out = self.dynamic_eval(self.train_unrolls,
                                         self.test_inputs,
                                         self.matrix_invs_test,
                                         self.M_tensor_test,
                                         self.q_mat_test)
            test_loss, test_out, time_per_prob = eval_out
        self.te_losses.append(test_loss)

        time_per_iter = time_per_prob / self.train_unrolls
        return test_loss, time_per_iter

    def evaluate(self, k, inputs, matrix_inv, M, q, z_stars, fixed_ws, tag='test'):
        if self.static_flag:
            return self.static_eval(k, inputs, q, z_stars, tag=tag, fixed_ws=fixed_ws)
        else:
            return self.dynamic_eval(k, inputs, matrix_inv, M, q, tag=tag, fixed_ws=fixed_ws)

    def static_eval(self, k, inputs, q, z_stars, tag='test', fixed_ws=False):
        if fixed_ws:
            curr_loss_fn = self.loss_fn_fixed_ws
        else:
            curr_loss_fn = self.loss_fn_eval
        num_probs, _ = inputs.shape
        test_time0 = time.time()

        loss, out = curr_loss_fn(
            self.params, inputs, q, k, z_stars)
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

    def create_loss_fn(self, input_dict):
        bypass_nn = input_dict['bypass_nn']
        diff_required = input_dict['diff_required']
        supervised = input_dict['supervised']
        proj, n, normalize_alpha = self.proj, self.n, self.normalize_alpha
        M_static, factor_static = self.static_M, self.static_algo_factor
        share_all, Z_shared = self.share_all, self.Z_shared
        loss_method, static_flag = self.loss_method, self.static_flag
        # partial_fixed_point = partial(fixed_point, proj=proj)

        def predict(params, input, q, iters, z_star, factor, M):
            P, A = M[:n, :n], -M[n:, :n]
            b, c = q[n:], q[:n]
            alpha = None

            if bypass_nn:
                z = input
            else:
                if share_all:
                    alpha_raw = predict_y(params, input)
                    alpha = normalize_alpha_fn(alpha_raw, normalize_alpha)
                    z = Z_shared @ alpha
                else:
                    nn_output = predict_y(params, input)
                    z = nn_output
            z0 = z
            iter_losses = jnp.zeros(iters)
            primal_residuals, dual_residuals = jnp.zeros(iters), jnp.zeros(iters)

            # suppose iters = 500
            # then we store u_1, ..., u_500 and z_0, z_1, ..., z_500
            all_u, all_z = jnp.zeros((iters, z.size)), jnp.zeros((iters, z.size))
            all_z_ = jnp.zeros((iters + 1, z.size))
            all_z_ = all_z_.at[0, :].set(z)

            if diff_required:
                fp_train_partial = partial(fp_train, q=q, factor=factor,
                                           supervised=supervised, z_star=z_star, proj=proj)
                val = z, iter_losses
                out = jax.lax.fori_loop(0, iters, fp_train_partial, val)
                z_final, iter_losses = out
            else:
                # def fp_eval(i, val):
                #     z, z_prev, loss_vec, all_z, all_u, primal_residuals, dual_residuals = val
                #     z_next, u, u_tilde, v = partial_fixed_point(z, q, factor)
                #     diff = jnp.linalg.norm(z_next - z)
                #     loss_vec = loss_vec.at[i].set(diff)

                #     # primal and dual residuals
                #     pr = jnp.linalg.norm(A @ u[:n] + v[n:] - b)
                #     dr = jnp.linalg.norm(A.T @ u[n:] + P @ u[:n] + c)
                #     primal_residuals = primal_residuals.at[i].set(pr)
                #     dual_residuals = dual_residuals.at[i].set(dr)

                #     all_z = all_z.at[i, :].set(z)
                #     all_u = all_u.at[i, :].set(u)
                #     return z_next, z_prev, loss_vec, all_z, all_u, primal_residuals, dual_residuals
                fp_eval_partial = partial(fp_eval, q=q, factor=factor,
                                          proj=proj, P=P, A=A, c=c, b=b)
                val = z, z, iter_losses, all_z, all_u, primal_residuals, dual_residuals
                out = jax.lax.fori_loop(0, iters, fp_eval_partial, val)
                z_final, z_penult, iter_losses, all_z, all_u, primal_residuals, dual_residuals = out
                all_z_ = all_z_.at[1:, :].set(all_z)

                # do the angle(z^{k+1} - z^k, z^k - z^{k-1})
                diffs = jnp.diff(all_z_, axis=0)
                angles = batch_angle(diffs[:-1], diffs[1:])

            loss = final_loss(loss_method, z_final, iter_losses, supervised, z0, z_star)
            out = all_z_, z_final, alpha, all_u
            if diff_required:
                return loss
            else:
                return loss, iter_losses, angles, primal_residuals, dual_residuals, out
        loss_fn = predict_2_loss(predict, static_flag, diff_required, factor_static, M_static)
        return loss_fn


def compute_angle(d1, d2):
    cos = d1 @ d2 / (jnp.linalg.norm(d1) * jnp.linalg.norm(d2))
    angle = jnp.arccos(cos)
    return angle


batch_angle = vmap(compute_angle, in_axes=(0, 0), out_axes=(0))


def final_loss(loss_method, z_last, iter_losses, supervised, z0, z_star):
    """
    encodes several possible loss functions

    z_last is the last iterate from DR splitting
    z_penultimate is the second to last iterate from DR splitting

    z_star is only used if supervised

    z0 is only used if the loss_method is first_2_last
    """
    if supervised:
        if loss_method == 'constant_sum':
            loss = iter_losses.sum()
        elif loss_method == 'fixed_k':
            loss = jnp.linalg.norm(z_last - z_star)
    else:
        if loss_method == 'increasing_sum':
            weights = (1+jnp.arange(iter_losses.size))
            loss = iter_losses @ weights
        elif loss_method == 'constant_sum':
            loss = iter_losses.sum()
        elif loss_method == 'fixed_k':
            loss = iter_losses[-1]
        elif loss_method == 'first_2_last':
            loss = jnp.linalg.norm(z_last - z0)
    return loss


def normalize_alpha_fn(alpha, normalize_alpha):
    """
    normalizes the alpha vector according to the method prescribed
        in the normalize_alpha input
    """
    if normalize_alpha == 'conic':
        alpha = jnp.maximum(0, alpha)
    elif normalize_alpha == 'sum':
        alpha = alpha / alpha.sum()
    elif normalize_alpha == 'convex':
        alpha = jnp.maximum(0, alpha)
        alpha = alpha / alpha.sum()
    elif normalize_alpha == 'softmax':
        alpha = jax.nn.softmax(alpha)
    return alpha


def predict_2_loss(predict, static_flag, diff_required, factor_static, M_static):
    """
    given the predict fn this returns the loss fn

    basically breaks the prediction fn into multiple cases
        - diff_required used for training, but not evaluation
        - static_flag is True if the matrices (P, A) are the same for each problem
            factor_static and M_static are shared for all problems and passed in

    for the evaluation, we store a lot more information
    for the training, we store nothing - just return the loss
        this could be changed - need to turn the boolean has_aux=True
            in self.optimizer = OptaxSolver(opt=optax.adam(
                self.lr), fun=self.loss_fn_train, has_aux=False)

    in all forward passes, the number of iterations is static
    """
    if diff_required:
        # out_axes for (loss)
        out_axes = (0)
    else:
        # out_axes for
        #   (loss, iter_losses, angles, primal_residuals, dual_residuals, out)
        #   out = (all_z_, z_next, alpha, all_u)
        out_axes = (0, 0, 0, 0, 0, (0, 0, 0, 0))
    if static_flag:
        predict_final = partial(predict,
                                factor=factor_static,
                                M=M_static
                                )
        batch_predict = vmap(predict_final, in_axes=(
            None, 0, 0, None, 0), out_axes=out_axes)

        @partial(jit, static_argnums=(3,))
        def loss_fn(params, inputs, q, iters, z_stars):
            if diff_required:
                losses = batch_predict(params, inputs, q, iters, z_stars)
                return losses.mean()
            else:
                predict_out = batch_predict(
                    params, inputs, q, iters, z_stars)
                losses, iter_losses, angles, primal_residuals, dual_residuals, out = predict_out
                loss_out = out, losses, iter_losses, angles, primal_residuals, dual_residuals
                return losses.mean(), loss_out
    else:
        batch_predict = vmap(predict, in_axes=(
            None, 0, 0, None, 0, 0), out_axes=out_axes)

        @partial(jax.jit, static_argnums=(5,))
        def loss_fn(params, inputs, factor, M, q, iters):
            if diff_required:
                losses = batch_predict(params, inputs, q, iters, factor, M)
                return losses.mean()
            else:
                predict_out = batch_predict(
                    params, inputs, q, iters, factor, M)
                losses, iter_losses, angles, primal_residuals, dual_residuals, out = predict_out
                loss_out = out, losses, iter_losses, angles, primal_residuals, dual_residuals
                return losses.mean(), loss_out
    return loss_fn


def fp_train(i, val, q, factor, supervised, z_star, proj):
    z, loss_vec = val
    z_next, u, u_tilde, v = fixed_point(z, q, factor, proj)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fp_eval(i, val, q, factor, proj, P, A, c, b):
    n = c.size
    z, z_prev, loss_vec, all_z, all_u, primal_residuals, dual_residuals = val
    z_next, u, u_tilde, v = fixed_point(z, q, factor, proj)
    diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)

    # primal and dual residuals
    pr = jnp.linalg.norm(A @ u[:n] + v[n:] - b)
    dr = jnp.linalg.norm(A.T @ u[n:] + P @ u[:n] + c)
    primal_residuals = primal_residuals.at[i].set(pr)
    dual_residuals = dual_residuals.at[i].set(dr)

    all_z = all_z.at[i, :].set(z)
    all_u = all_u.at[i, :].set(u)
    return z_next, z_prev, loss_vec, all_z, all_u, primal_residuals, dual_residuals
