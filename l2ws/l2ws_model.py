import logging
import time
from functools import partial

import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, random, vmap
from jax.config import config
from jaxopt import OptaxSolver

from l2ws.algo_steps import create_eval_fn, create_train_fn, lin_sys_solve
from l2ws.utils.nn_utils import init_network_params, predict_y

# from l2ws.scs_model import SCSmodel
# from l2ws.scs_model import SCSmodel
config.update("jax_enable_x64", True)


class L2WSmodel(object):
    # def __init__(self, dict):
    def __init__(self, 
                 train_unrolls=5,
                 train_inputs=None,
                 test_inputs=None,
                 regression=False,
                 nn_cfg={},
                 plateau_decay={},
                 jit=True,
                 eval_unrolls=500,
                 z_stars_train=None,
                 z_stars_test=None,
                 x_stars_train=None,
                 x_stars_test=None,
                 y_stars_train=None,
                 y_stars_test=None,
                 loss_method='fixed_k',
                 algo_dict={}):
        dict = algo_dict

        # essential pieces for the model
        self.initialize_essentials(jit, eval_unrolls, train_unrolls, train_inputs, test_inputs)

        # set defaults
        self.set_defaults()

        # initialize algorithm specifics
        self.initialize_algo(dict)

        # post init changes
        self.post_init_changes()

        # optimal solutions (not needed as input)
        # self.setup_optimal_solutions(dict)
        self.setup_optimal_solutions(z_stars_train, z_stars_test, x_stars_train, x_stars_test, 
                                     y_stars_train, y_stars_test)

        # create_all_loss_fns
        self.create_all_loss_fns(loss_method, regression)
        # to describe the final loss function (not the end-to-end loss fn)
        # self.loss_method = dict.get('loss_method', 'fixed_k')
        # self.supervised = dict.get('supervised', False)
        # self.create_all_loss_fns(dict)

        # neural network setup
        # self.initialize_neural_network(dict)
        self.initialize_neural_network(nn_cfg, plateau_decay)

        # init to track training
        self.init_train_tracking()

    
    def post_init_changes(self):
        if not hasattr(self, 'q_mat_train'):
            self.q_mat_train = self.theta_mat_train
        if not hasattr(self, 'q_mat_test'):
            self.q_mat_test = self.theta_mat_test


    def set_defaults(self):
        # unless turned off in the subclass, these are the default settings
        self.factors_required = False
        self.factor_static = None


    # def initialize_essentials(self, input_dict):
    def initialize_essentials(self, jit, eval_unrolls, train_unrolls, train_inputs, test_inputs):
        self.jit = jit
        self.eval_unrolls = eval_unrolls
        self.train_unrolls = train_unrolls + 1
        self.train_inputs, self.test_inputs = train_inputs, test_inputs
        self.N_train, self.N_test = self.train_inputs.shape[0], self.test_inputs.shape[0]
        # self.share_all = input_dict.get('share_all', False)
        self.batch_angle = vmap(self.compute_angle, in_axes=(0, 0), out_axes=(0))
        self.static_flag = True

    # def setup_optimal_solutions(self, dict):
    def setup_optimal_solutions(self, z_stars_train, z_stars_test, x_stars_train=None, 
                                x_stars_test=None, y_stars_train=None, y_stars_test=None):
        # if dict.get('z_stars_train', None) is not None:
        if z_stars_train is not None:
            self.z_stars_train = jnp.array(z_stars_train) # jnp.array(dict['z_stars_train'])
            self.z_stars_test = jnp.array(z_stars_test) # jnp.array(dict['z_stars_test'])
        else:
            self.z_stars_train, self.z_stars_test = None, None

    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = self.supervised and diff_required
        loss_method = self.loss_method

        def predict(params, input, q, iters, z_star, factor):
            if self.algo == 'scs':
                # q = lin_sys_solve(self.factor, q)
                q = lin_sys_solve(factor, q)
            else:
                pass
            # z0, alpha = self.predict_warm_start(params, input, bypass_nn, hsde=hsde)
            z0 = self.predict_warm_start(params, input, bypass_nn)

            # if self.out_axes_length == 8:
            # if isinstance(self, SCSmodel):
            #     q = lin_sys_solve(self.factor, q)

            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn

            if diff_required:
                if self.factors_required:
                    z_final, iter_losses = train_fn(k=iters,
                                                    z0=z0,
                                                    q=q,
                                                    supervised=supervised,
                                                    z_star=z_star,
                                                    factor=factor)
                else:
                    z_final, iter_losses = train_fn(k=iters,
                                                    z0=z0,
                                                    q=q,
                                                    supervised=supervised,
                                                    z_star=z_star)
            else:
                if self.factors_required:
                    eval_out = eval_fn(k=iters,
                                       z0=z0,
                                       q=q,
                                       factor=factor,
                                       supervised=supervised,
                                       z_star=z_star)
                else:
                    eval_out = eval_fn(k=iters,
                                       z0=z0,
                                       q=q,
                                       supervised=supervised,
                                       z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]

                # compute angle(z^{k+1} - z^k, z^k - z^{k-1})
                # diffs = jnp.diff(z_all_plus_1, axis=0)
                # angles = self.batch_angle(diffs[:-1], diffs[1:])
                angles = None

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
            # batch_factors = self.factors[batch_indices, :, :]
            batch_factors = (self.factors_train[0][batch_indices,
                             :, :], self.factors_train[1][batch_indices, :])
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

    def evaluate(self, k, inputs, b, z_stars, fixed_ws, factors=None, tag='test', light=False):
        if self.factors_required and not self.factor_static_bool:
            return self.dynamic_eval(k, inputs, b, z_stars, 
                                     factors=factors, tag=tag, fixed_ws=fixed_ws)
        else:
            return self.static_eval(k, inputs, b, z_stars, tag=tag, fixed_ws=fixed_ws, light=light)

    def short_test_eval(self):
        # z_stars_test = self.z_stars_test if self.supervised else None
        z_stars_test = self.z_stars_test

        if self.factors_required and not self.factor_static_bool:
            test_loss, test_out, time_per_prob = self.dynamic_eval(self.train_unrolls,
                                                                   self.test_inputs,
                                                                   self.q_mat_test,
                                                                   z_stars_test,
                                                                   factors=self.factors_test)
        else:
            test_loss, test_out, time_per_prob = self.static_eval(self.train_unrolls,
                                                                  self.test_inputs,
                                                                  self.q_mat_test,
                                                                  z_stars_test)

        self.te_losses.append(test_loss)

        time_per_iter = time_per_prob / self.train_unrolls
        return test_loss, time_per_iter

    def dynamic_eval(self, k, inputs, b, z_stars, factors, tag='test', fixed_ws=False):
        if fixed_ws:
            curr_loss_fn = self.loss_fn_fixed_ws
        else:
            curr_loss_fn = self.loss_fn_eval
        num_probs, _ = inputs.shape

        test_time0 = time.time()

        loss, out = curr_loss_fn(self.params, inputs, b, k, z_stars, factors)
        time_per_prob = (time.time() - test_time0)/num_probs

        return loss, out, time_per_prob

    def static_eval(self, k, inputs, b, z_stars, tag='test', fixed_ws=False, light=False):
        # if light:
        #     if fixed_ws:
        #         curr_loss_fn = self.loss_fn_fixed_ws_light
        #     else:
        #         curr_loss_fn = self.loss_fn_train
        #     num_probs, _ = inputs.shape

        #     test_time0 = time.time()

        #     loss, out = curr_loss_fn(self.params, inputs, b, k, z_stars)
        #     time_per_prob = (time.time() - test_time0)/num_probs

        #     return loss, out, time_per_prob
        # else:
        #     if fixed_ws:
        #         curr_loss_fn = self.loss_fn_eval
        #     else:
        #         curr_loss_fn = self.loss_fn_fixed_ws
        #     num_probs, _ = inputs.shape

        #     test_time0 = time.time()

        #     loss, out = curr_loss_fn(self.params, inputs, b, k, z_stars)
        #     time_per_prob = (time.time() - test_time0)/num_probs

        #     return loss, out, time_per_prob
        if fixed_ws:
            curr_loss_fn = self.loss_fn_fixed_ws
        else:
            curr_loss_fn = self.loss_fn_eval
        num_probs, _ = inputs.shape

        test_time0 = time.time()

        loss, out = curr_loss_fn(self.params, inputs, b, k, z_stars)
        time_per_prob = (time.time() - test_time0)/num_probs

        return loss, out, time_per_prob

    # def initialize_neural_network(self, input_dict):
    def initialize_neural_network(self, nn_cfg, plateau_decay):
        # nn_cfg = input_dict.get('nn_cfg', {})


        # neural network
        self.epochs, self.lr = nn_cfg.get('epochs', 10), nn_cfg.get('lr', 1e-3)
        self.decay_lr, self.min_lr = nn_cfg.get('decay_lr', False), nn_cfg.get('min_lr', 1e-7)

        # auto-decay learning rate
        # self.plateau_decay = input_dict.get('plateau_decay')
        self.plateau_decay = plateau_decay

        if self.plateau_decay is None:
            self.plateau_decay = dict(min_lr=1e-7, decay_factor=5,
                                      avg_window_size=50, tolerance=1e-2, patience=2)

        self.dont_decay_until = 2 * self.plateau_decay.get('avg_window_size', 10)
        self.epoch_decay_points = []

        # batching
        batch_size = nn_cfg.get('batch_size', self.N_train)
        self.batch_size = min([batch_size, self.N_train])
        self.num_batches = int(self.N_train/self.batch_size)

        # layer sizes
        input_size = self.train_inputs.shape[1]
        # if self.share_all:
        #     output_size = self.num_clusters
        # else:
        #     output_size = self.output_size
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
        # self.state = self.optimizer.init_state(self.params)

        # Initialize state with first elements of training data as inputs
        batch_indices = jnp.arange(self.N_train)
        input_init = self.train_inputs[batch_indices, :]
        q_init = self.q_mat_train[batch_indices, :]
        z_stars_init = self.z_stars_train[batch_indices, :] if self.supervised else None

        if self.factors_required and not self.factor_static_bool:
            batch_factors = (self.factors_train[0][batch_indices, :, :], 
                             self.factors_train[1][batch_indices, :])
            self.state = self.optimizer.init_state(init_params=self.params,
                                                   inputs=input_init,
                                                   b=q_init,
                                                   iters=self.train_unrolls,
                                                   z_stars=z_stars_init,
                                                   factors=batch_factors)
        else:
            self.state = self.optimizer.init_state(init_params=self.params,
                                                   inputs=input_init,
                                                   b=q_init,
                                                   iters=self.train_unrolls,
                                                   z_stars=z_stars_init)

    # def setup_share_all(self, dict):
    #     if self.share_all:
    #         self.num_clusters = dict.get('num_clusters', 10)
    #         self.pretrain_alpha = dict.get('pretrain_alpha', False)
    #         self.normalize_alpha = dict.get('normalize_alpha', 'none')
    #         out = self.cluster_z()
    #         self.Z_shared = out[0]
    #         self.train_cluster_indices, self.test_cluster_indices = out[1], out[2]
    #         self.X_list, self.Y_list = [], []
    #         if self.pretrain_alpha:
    #             self.pretrain_alphas(1000, None, share_all=True)

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

    # def create_all_loss_fns(self, dict):
    def create_all_loss_fns(self, loss_method, supervised):
        # to describe the final loss function (not the end-to-end loss fn)
        self.loss_method = loss_method
        self.supervised = supervised

        if not hasattr(self, 'train_fn') and not hasattr(self, 'k_steps_train_fn'):
            train_fn = create_train_fn(self.fixed_point_fn)
            eval_fn = create_eval_fn(self.fixed_point_fn)
            self.train_fn = partial(train_fn, jit=self.jit)
            self.eval_fn = partial(eval_fn, jit=self.jit)

        if not hasattr(self, 'train_fn'):
            self.train_fn = self.k_steps_train_fn
            self.eval_fn = self.k_steps_eval_fn

        e2e_loss_fn = self.create_end2end_loss_fn

        # end-to-end loss fn for training
        self.loss_fn_train = e2e_loss_fn(bypass_nn=False, diff_required=True)

        # end-to-end loss fn for evaluation
        self.loss_fn_eval = e2e_loss_fn(bypass_nn=False, diff_required=False)

        # end-to-end added fixed warm start eval - bypasses neural network
        self.loss_fn_fixed_ws = e2e_loss_fn(bypass_nn=True, diff_required=False)

        # end-to-end loss fn for evaluation of fixed ws - meant for light mode
        # self.loss_fn_fixed_ws_light = e2e_loss_fn(bypass_nn=True, diff_required=True)

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

    def predict_warm_start(self, params, input, bypass_nn):
        """
        gets the warm-start
        bypass_nn means we ignore the neural network and set z0=input
        """
        if bypass_nn:
            z0 = input
        else:
            nn_output = predict_y(params, input)
            z0 = nn_output
        if self.algo == 'scs':
            z0_full = jnp.ones(z0.size + 1)
            z0_full = z0_full.at[:z0.size].set(z0)
        else:
            z0_full = z0
        return z0_full

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
            # if self.out_axes_length is None:
            if hasattr(self, 'out_axes_length'):
                out_axes = (0,) * self.out_axes_length
            else:
                out_axes = (0,) * 4
        return out_axes

    def predict_2_loss(self, predict, diff_required):
        out_axes = self.get_out_axes_shape(diff_required)

        # just for reference, the arguments for predict are
        #   predict(params, input, q, iters, z_star, factor)

        if self.factors_required and not self.factor_static_bool:
            # for the case where the factors change for each problem
            # batch_predict = vmap(predict,
            #                     in_axes=(None, 0, 0, 0, None, 0),
            #                     out_axes=out_axes)
            batch_predict = vmap(predict,
                                 in_axes=(None, 0, 0, None, 0, (0, 0)),
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
            #   2. factor is constant for all problems (pass in the same factor as static argument)
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
