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
    predict_y, init_matrix_params
from utils.generic_utils import vec_symm, unvec_symm
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pd
from jax.config import config
from scipy.spatial import distance_matrix
import logging
config.update("jax_enable_x64", True)


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
        self.learn_XY = dict.get('learn_XY')
        self.num_clusters = dict.get('num_clusters')
        self.x_psd_indices = dict.get('x_psd_indices')
        self.y_psd_indices = dict.get('y_psd_indices')
        self.loss_method = dict.get('loss_method')
        self.share_all = dict.get('share_all')
        self.pretrain_alpha = dict.get('pretrain_alpha')
        self.normalize_alpha = dict.get('normalize_alpha')
        self.plateau_decay = dict.get('plateau_decay')
        self.dont_decay_until = 2 * self.plateau_decay.avg_window_size

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
        self.batched_predict_y = vmap(predict_y, in_axes=(None, 0))

        if self.share_all:
            output_size = self.num_clusters
        else:
            if self.prediction_variable == 'w':
                if self.psd:
                    n_x_non_psd = self.n - int(self.psd_size * (self.psd_size + 1) / 2)
                    n_y_non_psd = self.m - int(self.psd_size * (self.psd_size + 1) / 2)
                    n_x_low = n_x_non_psd + self.dx * self.psd_size
                    n_y_low = n_y_non_psd + self.dy * self.psd_size

                    output_size = n_x_low + n_y_low + self.tx + self.ty
                else:
                    output_size = self.n + self.m
            elif self.prediction_variable == 'x':
                output_size = self.n
        layer_sizes = [input_size] + \
            self.nn_cfg['intermediate_layer_sizes'] + [output_size]

        self.nn_params = init_network_params(layer_sizes, random.PRNGKey(0))
        key = 0
        if self.share_all:
            out = self.cluster_z()
            Z_shared = out[0]
            self.train_cluster_indices, self.test_cluster_indices = out[1], out[2]
            self.X_list, self.Y_list = [], []
            self.params = self.nn_params
            if self.pretrain_alpha:
                self.pretrain_alphas(1000, None, share_all=True)

        else:
            Z_shared = None
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
                    # self.Y_list = cluster_init_Y_list()
                    self.params = self.nn_params
                    if self.pretrain_alpha:
                        self.pretrain_alphas(1000, n_x_low + n_y_low)
            else:
                self.X_list, self.Y_list = [], []
                self.params = self.nn_params
        # self.state = None

        self.epoch = 0

        train_loss_dict = {'static_flag': self.static_flag,
                           'lin_sys_solve': lin_sys_solve,
                           'proj': proj,
                           'unrolls': self.train_unrolls,
                           'm': self.m,
                           'n': self.n,
                           'prediction_variable': self.prediction_variable,
                           'M_static': self.static_M,
                           'factor_static': self.static_algo_factor,
                           'diff_required': True,
                           'angle_anchors': self.angle_anchors,
                           'supervised': self.supervised,
                           'num_nn_params': len(self.nn_params),
                           'tx': self.tx,
                           'ty': self.ty,
                           'low_2_high_dim': self.low_2_high_dim,
                           'X_list_fixed': self.X_list,
                           'Y_list_fixed': self.Y_list,
                           'learn_XY': self.learn_XY,
                           'loss_method': self.loss_method,
                           'share_all': self.share_all,
                           'Z_shared': Z_shared,
                           'normalize_alpha': self.normalize_alpha
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
                          'diff_required': False,
                          'angle_anchors': self.angle_anchors,
                          'supervised': self.supervised,
                          'num_nn_params': len(self.nn_params),
                          'tx': self.tx,
                          'ty': self.ty,
                          'low_2_high_dim': self.low_2_high_dim,
                          'X_list_fixed': self.X_list,
                          'Y_list_fixed': self.Y_list,
                          'learn_XY': self.learn_XY,
                          'loss_method': self.loss_method,
                          'share_all': self.share_all,
                          'Z_shared': Z_shared,
                          'normalize_alpha': self.normalize_alpha
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
                         'diff_required': False,
                         'angle_anchors': self.angle_anchors,
                         'supervised': self.supervised,
                         'num_nn_params': len(self.nn_params),
                         'tx': self.tx,
                         'ty': self.ty,
                         'low_2_high_dim': self.low_2_high_dim,
                         'X_list_fixed': self.X_list,
                         'Y_list_fixed': self.Y_list,
                         'learn_XY': self.learn_XY,
                         'loss_method': self.loss_method,
                         'share_all': self.share_all,
                         'Z_shared': Z_shared,
                         'normalize_alpha': self.normalize_alpha
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

    def pretrain_alphas(self, num_iters, n_xy_low, share_all=False, stepsize=.001, method='adam', batches=10):
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

        '''
        do a 1-hot encoding - assume given vector of indices
        '''
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
            # if self.tx + self.ty == -1:
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

            # z = M @ uu + uu + q
            # pretrain_loss = jnp.mean(jnp.sum((z - targets)**2, axis=1))
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
        params = self.params  # self.nn_params
        pretrain_losses = np.zeros(batches + 1)
        pretrain_test_losses = np.zeros(batches + 1)

        if self.prediction_variable == 'w':
            train_targets = self.u_stars_train
            test_targets = self.u_stars_test
            # train_targets = self.w_stars_train
            # test_targets = self.w_stars_test
        elif self.prediction_variable == 'x':
            train_targets = self.x_stars_train
            test_targets = self.x_stars_test

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
        self.avg_window_plateau: take the last avg_window number of epochs and compared it against the previous
            avg_window number of epochs to compare
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
                        self.lr), fun=self.loss_fn_train, has_aux=True)
                elif self.nn_cfg.method == 'sgd':
                    self.optimizer = OptaxSolver(opt=optax.sgd(
                        self.lr), fun=self.loss_fn_train, has_aux=True)
                self.state = self.optimizer.init_state(self.params)
                logging.info(f"the decay rate is now {self.lr}")

                # don't decay for another 2 * window number of epochs
                self.dont_decay_until = self.epoch + 2 * patience * self.plateau_decay.avg_window_size

    def train_batch(self, batch_indices, decay_lr_flag=False, writer=None, logf=None):
        batch_inputs = self.train_inputs[batch_indices, :]
        batch_q_data = self.q_mat_train[batch_indices, :]
        batch_z_stars = self.w_stars_train[batch_indices, :]

        # check if we need to update lr
        # if decay_lr_flag:
        #     if self.min_lr <= self.lr * self.decay_lr and self.decay_lr < 1.0:
        #         # re-initialize the optimizer
        #         self.lr = self.lr * self.decay_lr
        #         print(f"lr decayed to {self.lr}")
        #         self.optimizer = OptaxSolver(opt=optax.adam(
        #             self.lr), fun=self.loss_fn_train, has_aux=True)
        #         self.state = self.optimizer.init_state(self.params)

        t0 = time.time()
        results = self.optimizer.update(params=self.params,
                                        state=self.state,
                                        inputs=batch_inputs,
                                        q=batch_q_data,
                                        iters=self.train_unrolls,
                                        z_stars=batch_z_stars)

        self.params, self.state = results

        t1 = time.time()
        time_per_batch = (t1 - t0)
        print('time per batch', time_per_batch)
        train_out = self.state.aux

        print(
            f"[Step {self.state.iter_num}] train loss: {self.state.value:.6f}")

        row = np.array([self.state.value])
        self.train_data.append(pd.Series(row))
        self.tr_losses_batch.append(self.state.value)
        # self.te_losses.append(test_loss)
        last10 = np.array(self.tr_losses_batch[-10:])
        moving_avg_train = last10.mean()
        return self.state.iter_num, self.state.value, moving_avg_train

    def short_test_eval(self, writer=None, logf=None):
        if self.static_flag:
            test_loss, test_out, time_per_prob = self.evaluate(self.train_unrolls,
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
        if writer is not None:
            writer.writerow({
                'iter': self.state.iter_num,
                'train_loss': self.state.value,
                # 'moving_avg_train': moving_avg_train,
                'test_loss': test_loss,
                'time_per_iter': time_per_iter
            })
            logf.flush()

    # def evaluate(self, k, inputs, q, tag='test', fixed_ws=False):

    def evaluate(self, k, inputs, q, z_stars, tag='test', fixed_ws=False):
        if fixed_ws:
            curr_loss_fn = self.loss_fn_fixed_ws
        else:
            curr_loss_fn = self.loss_fn_eval
        num_probs, _ = inputs.shape
        test_time0 = time.time()
        # loss, out = curr_loss_fn(
        #     self.params, inputs, q, k)
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
    # unrolls = input_dict['unrolls']
    m, n = input_dict['m'], input_dict['n']
    prediction_variable = input_dict['prediction_variable']
    diff_required = input_dict['diff_required']
    angle_anchors = input_dict['angle_anchors']
    supervised = input_dict['supervised']
    num_nn_params = input_dict['num_nn_params']
    tx, ty = input_dict['tx'], input_dict['ty']
    low_2_high_dim = input_dict['low_2_high_dim']
    learn_XY = input_dict['learn_XY']
    X_list_fixed, Y_list_fixed = input_dict['X_list_fixed'], input_dict['Y_list_fixed']
    loss_method = input_dict['loss_method']
    share_all = input_dict['share_all']
    Z_shared = input_dict['Z_shared']
    normalize_alpha = input_dict['normalize_alpha']

    # if dynamic, the next 2 set to None
    M_static, factor_static = input_dict['M_static'], input_dict['factor_static']

    def fixed_point(z_init, factor, q):
        u_tilde = lin_sys_solve(factor, z_init - q)
        u_temp = (2*u_tilde - z_init)
        u = proj(u_temp)
        v = u + z_init - 2*u_tilde
        z = z_init + u - u_tilde
        return z, u, v

    def predict(params, input, q, iters, z_star, factor, M):
        P, A = M[:n, :n], -M[n:, :n]
        b, c = q[n:], q[:n]
        alpha = None
        if prediction_variable == 'w':
            if share_all:
                alpha = predict_y(params, input)
                if normalize_alpha == 'conic':
                    alpha = jnp.maximum(0, alpha)
                elif normalize_alpha == 'sum':
                    alpha = alpha / alpha.sum()
                elif normalize_alpha == 'convex':
                    alpha = jnp.maximum(0, alpha)
                    alpha = alpha / alpha.sum()
                z = Z_shared @ alpha
                u_ws = z
            else:
                # if tx + ty == -1:
                if tx is None or ty is None:
                    nn_output = predict_y(params, input)
                    u_ws = nn_output
                else:
                    nn_params = params[:num_nn_params]
                    nn_output = predict_y(nn_params, input)
                    if learn_XY:
                        X_list = params[num_nn_params:num_nn_params + tx]
                        Y_list = params[num_nn_params + tx:]
                    else:
                        X_list = X_list_fixed
                        Y_list = Y_list_fixed
                    u_ws = low_2_high_dim(nn_output, X_list, Y_list)

                # z = M @ u_ws + u_ws + q
                z = u_ws
        elif prediction_variable == 'x':
            z = input
            u_ws = input
        z0 = z
        iter_losses = jnp.zeros(iters)
        primal_residuals = jnp.zeros(iters)
        dual_residuals = jnp.zeros(iters)
        # angles = jnp.zeros((len(angle_anchors), iters-1))
        angles = jnp.zeros((len(angle_anchors) + 1, iters-1))
        all_u = jnp.zeros((iters, z.size))
        all_z = jnp.zeros((iters, z.size))
        all_z_ = jnp.zeros((iters, z.size))
        all_z_ = all_z_.at[0, :].set(z)

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
            def _fp_(i, val):
                z, z_prev, loss_vec, all_z, all_u, primal_residuals, dual_residuals = val
                z_next, u, v = fixed_point(z, factor, q)
                diff = jnp.linalg.norm(z_next - z)
                loss_vec = loss_vec.at[i].set(diff)

                pr = jnp.linalg.norm(A @ u[:n] + v[n:] - b)
                dr = jnp.linalg.norm(A.T @ u[n:] + P @ u[:n] + c)
                primal_residuals = primal_residuals.at[i].set(pr)
                dual_residuals = dual_residuals.at[i].set(dr)

                all_z = all_z.at[i, :].set(z)
                all_u = all_u.at[i, :].set(u)
                return z_next, z_prev, loss_vec, all_z, all_u, primal_residuals, dual_residuals
            val = z, z, iter_losses, all_z, all_u, primal_residuals, dual_residuals
            out = jax.lax.fori_loop(0, iters, _fp_, val)
            z, z_prev, iter_losses, all_z, all_u, primal_residuals, dual_residuals = out

            # do angles
            diffs = jnp.diff(all_z, axis=0)

            for j in range(len(angle_anchors)):
                d1 = diffs[angle_anchors[j], :]

                def compute_angle(d2):
                    cos = d1 @ d2 / (jnp.linalg.norm(d1) * jnp.linalg.norm(d2))
                    angle = jnp.arccos(cos)
                    return angle
                batch_angle = vmap(compute_angle, in_axes=(0), out_axes=(0))
                curr_angles = batch_angle(diffs)
                angles = angles.at[j, :].set(curr_angles)

            # do the angle(z^{k+1} - z^k, z^k - z^{k-1})
            diffs = jnp.diff(all_z, axis=0)

            def compute_angle(d1, d2):
                cos = d1 @ d2 / (jnp.linalg.norm(d1) * jnp.linalg.norm(d2))
                angle = jnp.arccos(cos)
                return angle
            batch_angle = vmap(compute_angle, in_axes=(0, 0), out_axes=(0))
            curr_angles = batch_angle(diffs[:-1], diffs[1:])

            angles = angles.at[-1, :-1].set(curr_angles)

        # unroll 1 more time for the loss
        u_tilde = lin_sys_solve(factor, z - q)
        u_temp = 2*u_tilde - z
        u = proj(u_temp)
        z_next = z + u - u_tilde

        if supervised:
            loss = jnp.linalg.norm(z_next - z_star)
        else:
            if loss_method == 'increasing_sum':
                weights = (1+jnp.arange(iter_losses.size))
                loss = iter_losses @ weights
            elif loss_method == 'constant_sum':
                loss = iter_losses.sum()
            elif loss_method == 'fixed_k':
                loss = jnp.linalg.norm(z_next - z)
            elif loss_method == 'first_2_last':
                loss = jnp.linalg.norm(z_next-z0)

        # out = x_primal, z_next, u, all_x_primals
        all_z_ = all_z_.at[1:, :].set(all_z[:-1, :])
        out = all_z_, z_next, alpha, all_u

        if diff_required:
            return loss, iter_losses, angles, out
        else:
            return loss, iter_losses, angles, primal_residuals, dual_residuals, out

    if diff_required:
        out_axes = (0, 0, 0, (0, 0, 0, 0))
    else:
        out_axes = (0, 0, 0, 0, 0, (0, 0, 0, 0))

    if static_flag:
        predict_final = functools.partial(predict,
                                          factor=factor_static,
                                          M=M_static
                                          )
        batch_predict = vmap(predict_final, in_axes=(
            None, 0, 0, None, 0), out_axes=out_axes)

        @functools.partial(jax.jit, static_argnums=(3,))
        def loss_fn(params, inputs, q, iters, z_stars):
            if diff_required:
                losses, iter_losses, angles, out = batch_predict(
                    params, inputs, q, iters, z_stars)
                loss_out = out, losses, iter_losses, angles
            else:
                losses, iter_losses, angles, primal_residuals, dual_residuals, out = batch_predict(
                    params, inputs, q, iters, z_stars)
                loss_out = out, losses, iter_losses, angles, primal_residuals, dual_residuals

            return losses.mean(), loss_out
    else:
        batch_predict = vmap(predict, in_axes=(
            None, 0, 0, None, 0, 0), out_axes=out_axes)

        @functools.partial(jax.jit, static_argnums=(5,))
        def loss_fn(params, inputs, factor, M, q, iters):
            if diff_required:
                losses, iter_losses, angles, out = batch_predict(
                    params, inputs, q, iters, factor, M)
                loss_out = out, losses, iter_losses, angles
            else:
                losses, iter_losses, angles, primal_residuals, dual_residuals, out = batch_predict(
                    params, inputs, q, iters, factor, M)
                loss_out = out, losses, iter_losses, angles, primal_residuals, dual_residuals
            return losses.mean(), loss_out

    return loss_fn
