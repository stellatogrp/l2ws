from jax import jit, vmap
import jax.numpy as jnp
import jax
from jax import random
import optax
import time
from jaxopt import OptaxSolver
from l2ws.utils.nn_utils import init_network_params, \
    predict_y, batched_predict_y
from l2ws.utils.generic_utils import unvec_symm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from jax.config import config
from scipy.spatial import distance_matrix
import logging
# from l2ws.algo_steps import k_steps_train, k_steps_eval, lin_sys_solve, k_steps_train_ista, k_steps_eval_ista
from functools import partial
from l2ws.algo_steps import lin_sys_solve
# from l2ws.scs_model import SCSmodel
# from l2ws.scs_model import SCSmodel
config.update("jax_enable_x64", True)


class L2WSmodel(object):
    def __init__(self, dict):
        # essential pieces for the model
        self.initialize_essentials(dict)

        # initialize algorithm specifics
        self.initialize_algo(dict)

        # optimal solutions (not needed as input)
        self.setup_optimal_solutions(dict)

        # share all method
        # self.setup_share_all(dict)

        # create_all_loss_fns
        self.create_all_loss_fns(dict)

        # neural network setup
        self.initialize_neural_network(dict)

        # init to track training
        self.init_train_tracking()

    def initialize_essentials(self, input_dict):
        self.jit = input_dict.get('jit', True)
        self.eval_unrolls = input_dict.get('eval_unrolls', 500)
        self.train_unrolls = input_dict['train_unrolls']
        self.train_inputs, self.test_inputs = input_dict['train_inputs'], input_dict['test_inputs']
        self.N_train, self.N_test = self.train_inputs.shape[0], self.test_inputs.shape[0]
        self.share_all = input_dict.get('share_all', False)
        # self.algorithm = input_dict['algorithm']
        self.batch_angle = vmap(self.compute_angle, in_axes=(0, 0), out_axes=(0))
        self.static_flag = True

    def setup_optimal_solutions(self, dict):
        if dict.get('z_stars_train', None) is not None:
            self.z_stars_train = jnp.array(dict['z_stars_train'])
            self.z_stars_test = jnp.array(dict['z_stars_test'])
        else:
            self.z_stars_train, self.z_stars_test = None, None

    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = self.supervised and diff_required
        loss_method = self.loss_method

        def predict(params, input, q, iters, z_star, factor):
            if self.algo == 'scs':
                # q = lin_sys_solve(self.factor, q)
                q = lin_sys_solve(factor, q)
                hsde = self.hsde
            else:
                hsde = False
            z0, alpha = self.predict_warm_start(params, input, bypass_nn, hsde=hsde)

            # if self.out_axes_length == 8:
            # if isinstance(self, SCSmodel):
            #     q = lin_sys_solve(self.factor, q)

            if diff_required:
                z_final, iter_losses = self.k_steps_train_fn(k=iters, 
                                                             z0=z0, 
                                                             q=q,
                                                             supervised=supervised, 
                                                             z_star=z_star,
                                                             factor=factor)
            else:
                eval_out = self.k_steps_eval_fn(k=iters, 
                                                z0=z0, 
                                                q=q, 
                                                factor=factor, 
                                                supervised=supervised, 
                                                z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]

                # compute angle(z^{k+1} - z^k, z^k - z^{k-1})
                diffs = jnp.diff(z_all_plus_1, axis=0)
                angles = self.batch_angle(diffs[:-1], diffs[1:])

            loss = self.final_loss(loss_method, z_final, iter_losses, supervised, z0, z_star)

            if diff_required:
                return loss
            else:
                return_out = (loss, iter_losses, z_all_plus_1, angles) + eval_out[3:]
                return return_out
        # loss_fn = predict_2_loss(predict, static_flag, diff_required, factor_static, M_static)
        loss_fn = self.predict_2_loss(predict, diff_required)
        return loss_fn

    def train_batch(self, batch_indices, params, state):
        batch_inputs = self.train_inputs[batch_indices, :]
        batch_q_data = self.q_mat_train[batch_indices, :]
        batch_z_stars = self.z_stars_train[batch_indices, :] if self.supervised else None

        if self.factors_required and not self.factor_static_bool:
            # for only the case where the factors are needed
            batch_factors = self.factors[batch_indices, :, :]
            results = self.optimizer.update(params=params,
                                            state=state,
                                            inputs=batch_inputs,
                                            b=batch_q_data,
                                            iters=self.train_unrolls,
                                            z_stars=batch_z_stars,
                                            factors=batch_factors)
        else:
            # for either of the following cases
            #   1. factors needed, but are the same for all problems
            #   2. no factors are needed
            results = self.optimizer.update(params=params,
                                            state=state,
                                            inputs=batch_inputs,
                                            b=batch_q_data,
                                            iters=self.train_unrolls,
                                            z_stars=batch_z_stars)
        params, state = results
        return state.value, params, state

    def evaluate(self, k, inputs, b, z_stars, fixed_ws, tag='test'):
        return self.static_eval(k, inputs, b, z_stars, tag=tag, fixed_ws=fixed_ws)

    def short_test_eval(self):
        # z_stars_test = self.z_stars_test if self.supervised else None
        z_stars_test = self.z_stars_test
        
        test_loss, test_out, time_per_prob = self.static_eval(self.train_unrolls,
                                                              self.test_inputs,
                                                              self.q_mat_test,
                                                              z_stars_test)
        self.te_losses.append(test_loss)

        time_per_iter = time_per_prob / self.train_unrolls
        return test_loss, time_per_iter
    
    def static_eval(self, k, inputs, b, z_stars, tag='test', fixed_ws=False):
        if fixed_ws:
            curr_loss_fn = self.loss_fn_fixed_ws
        else:
            curr_loss_fn = self.loss_fn_eval
        num_probs, _ = inputs.shape

        test_time0 = time.time()

        loss, out = curr_loss_fn(self.params, inputs, b, k, z_stars)
        time_per_prob = (time.time() - test_time0)/num_probs

        return loss, out, time_per_prob

    def initialize_neural_network(self, input_dict):
        nn_cfg = input_dict.get('nn_cfg', {})

        # neural network
        self.epochs, self.lr = nn_cfg.get('epochs', 10), nn_cfg.get('lr', 1e-3)
        self.decay_lr, self.min_lr = nn_cfg.get('decay_lr', False), nn_cfg.get('min_lr', 1e-7)

        # auto-decay learning rate
        self.plateau_decay = input_dict.get('plateau_decay')
        if self.plateau_decay is None:
            self.plateau_decay = dict(min_lr=1e-7, decay_factor=5,
                                      avg_window_size=50, tolerance=1e-2, patience=2)

        self.dont_decay_until = 2 * self.plateau_decay['avg_window_size']
        self.epoch_decay_points = []

        # batching
        batch_size = nn_cfg.get('batch_size', self.N_train)
        self.batch_size = min([batch_size, self.N_train])
        self.num_batches = int(self.N_train/self.batch_size)

        # layer sizes
        input_size = self.train_inputs.shape[1]
        if self.share_all:
            output_size = self.num_clusters
        else:
            output_size = self.output_size
        hidden_layer_sizes = nn_cfg.get('intermediate_layer_sizes', [])

        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

        # initialize weights of neural network
        self.params = init_network_params(layer_sizes, random.PRNGKey(0))

        # initializes the optimizer
        self.optimizer_method = nn_cfg.get('method', 'adam')
        if self.optimizer_method == 'adam':
            self.optimizer = OptaxSolver(opt=optax.adam(
                self.lr), fun=self.loss_fn_train, has_aux=False)
        elif self.optimizer_method == 'sgd':
            self.optimizer = OptaxSolver(opt=optax.sgd(
                self.lr), fun=self.loss_fn_train, has_aux=False)
        self.state = self.optimizer.init_state(self.params)

    def setup_share_all(self, dict):
        if self.share_all:
            self.num_clusters = dict.get('num_clusters', 10)
            self.pretrain_alpha = dict.get('pretrain_alpha', False)
            self.normalize_alpha = dict.get('normalize_alpha', 'none')
            out = self.cluster_z()
            self.Z_shared = out[0]
            self.train_cluster_indices, self.test_cluster_indices = out[1], out[2]
            self.X_list, self.Y_list = [], []
            if self.pretrain_alpha:
                self.pretrain_alphas(1000, None, share_all=True)

    # def setup_optimal_solutions(self, dict):
    #     if dict.get('z_stars_train', None) is not None:
    #         self.y_stars_train, self.y_stars_test = dict['y_stars_train'], dict['y_stars_test']
    #         self.x_stars_train, self.x_stars_test = dict['x_stars_train'], dict['x_stars_test']
    #         self.z_stars_train = jnp.array(dict['z_stars_train'])
    #         self.z_stars_test = jnp.array(dict['z_stars_test'])
    #         self.u_stars_train = jnp.hstack([self.x_stars_train, self.y_stars_train])
    #         self.u_stars_test = jnp.hstack([self.x_stars_test, self.y_stars_test])
    #     else:
    #         self.z_stars_train, self.z_stars_test = None, None
    #     import pdb
    #     pdb.set_trace()

    # def create_end2end_loss_fn(self, bypass_nn, diff_required):
    #     raise NotImplementedError("Subclass needs to define this.")

    def create_all_loss_fns(self, dict):
        # to describe the final loss function (not the end-to-end loss fn)
        self.loss_method = dict.get('loss_method', 'fixed_k')
        self.supervised = dict.get('supervised', False)

        e2e_loss_fn = self.create_end2end_loss_fn

        # end-to-end loss fn for training
        self.loss_fn_train = e2e_loss_fn(bypass_nn=False, diff_required=True)

        # end-to-end loss fn for evaluation
        self.loss_fn_eval = e2e_loss_fn(bypass_nn=False, diff_required=False)

        # end-to-end added fixed warm start eval - bypasses neural network
        self.loss_fn_fixed_ws = e2e_loss_fn(bypass_nn=True, diff_required=False)

    def init_train_tracking(self):
        self.epoch = 0
        self.tr_losses = None
        self.te_losses = None
        self.train_data = []
        self.tr_losses_batch = []
        self.te_losses = []


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
        decay_factor = self.plateau_decay['decay_factor']

        window_batches = self.plateau_decay['avg_window_size'] * self.num_batches
        plateau_tolerance = self.plateau_decay['tolerance']
        patience = self.plateau_decay['patience']

        if self.plateau_decay['min_lr'] <= self.lr / decay_factor:
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
                if self.optimizer_method == 'adam':
                    self.optimizer = OptaxSolver(opt=optax.adam(
                        self.lr), fun=self.loss_fn_train, has_aux=False)
                elif self.optimizer_method == 'sgd':
                    self.optimizer = OptaxSolver(opt=optax.sgd(
                        self.lr), fun=self.loss_fn_train, has_aux=False)
                self.state = self.optimizer.init_state(self.params)
                logging.info(f"the decay rate is now {self.lr}")

                # log the current decay epoch
                self.epoch_decay_points.append(self.epoch)

                # don't decay for another 2 * window number of epochs
                wait_time = 2 * patience * self.plateau_decay['avg_window_size']
                self.dont_decay_until = self.epoch + wait_time

    def train_full_batch(self, params, state):
        """
        wrapper for train_batch where the batch size is N_train
        """
        batch_indices = jnp.arange(self.N_train)
        return self.train_batch(batch_indices, params, state)
    
    def predict_warm_start(self, params, input, bypass_nn, hsde=None, share_all=False, Z_shared=None, normalize_alpha=None):
        """
        gets the warm-start
        bypass_nn means we ignore the neural network and set z0=input
        """
        alpha = None
        if bypass_nn:
            z0 = input
        else:
            if share_all:
                alpha_raw = predict_y(params, input)
                alpha = self.normalize_alpha_fn(alpha_raw, normalize_alpha)
                z0 = Z_shared @ alpha
            else:
                nn_output = predict_y(params, input)
                z0 = nn_output
        if hsde:
            z0_full = jnp.ones(z0.size + 1)
            z0_full = z0_full.at[:z0.size].set(z0)
        else:
            z0_full = z0
        return z0_full, alpha


    def compute_angle(self, d1, d2):
        cos = d1 @ d2 / (jnp.linalg.norm(d1) * jnp.linalg.norm(d2))
        angle = jnp.arccos(cos)
        return angle


    def final_loss(self, loss_method, z_last, iter_losses, supervised, z0, z_star):
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
                # loss = jnp.linalg.norm(z_last[:-1]/z_star[-1] - z_star)
                loss = iter_losses[-1]
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
    
    def get_out_axes_shape(self, diff_required):
        if diff_required:
            # out_axes for (loss)
            out_axes = (0)
        else:
            # out_axes for
            #   (loss, iter_losses, angles, primal_residuals, dual_residuals, out)
            #   out = (all_z_, z_next, alpha, all_u, all_v)
            # out_axes = (0, 0, 0, 0)
            if self.out_axes_length is None:
                out_axes = (0,) * 4
            else:
                out_axes = (0,) * self.out_axes_length
        return out_axes
    
    def predict_2_loss(self, predict, diff_required):
        out_axes = self.get_out_axes_shape(diff_required)

        # just for reference, the arguments for predict are
        #   predict(params, input, q, iters, z_star, factor)

        if self.factors_required and not self.factor_static_bool:
            # for the case where the factors change for each problem
            batch_predict = vmap(predict,
                                in_axes=(None, 0, 0, 0, None, 0),
                                out_axes=out_axes)
            @partial(jit, static_argnums=(3,))
            def loss_fn(params, inputs, b, iters, z_stars, factors):
                if diff_required:
                    losses = batch_predict(params, inputs, b, iters, z_stars, factors)
                    return losses.mean()
                else:
                    predict_out = batch_predict(
                        params, inputs, b, iters, z_stars, factors)
                    losses = predict_out[0]
                    # loss_out = losses, iter_losses, angles, z_all
                    return losses.mean(), predict_out
        else:
            # for either of the following cases
            #   1. no factors are needed (pass in None as a static argument)
            #   2. factor is constant for all problems (pass in the same factor as a static argument)
            predict_partial = partial(predict, factor=self.factor_static)
            batch_predict = vmap(predict_partial,
                                in_axes=(None, 0, 0, None, 0),
                                out_axes=out_axes)
            @partial(jit, static_argnums=(3,))
            def loss_fn(params, inputs, b, iters, z_stars):
                if diff_required:
                    losses = batch_predict(params, inputs, b, iters, z_stars)
                    return losses.mean()
                else:
                    predict_out = batch_predict(
                        params, inputs, b, iters, z_stars)
                    losses = predict_out[0]
                    # loss_out = losses, iter_losses, angles, z_all
                    return losses.mean(), predict_out

        return loss_fn
    