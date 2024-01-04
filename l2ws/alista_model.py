from functools import partial

import jax.numpy as jnp
from jax import random

from l2ws.algo_steps import (
    k_steps_eval_alista,
    k_steps_train_alista,
)
from l2ws.l2ws_model import L2WSmodel
from l2ws.utils.nn_utils import calculate_total_penalty


class ALISTAmodel(L2WSmodel):
    def __init__(self, **kwargs):
        super(ALISTAmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'alista'
        self.factors_required = False
        self.q_mat_train, self.q_mat_test = input_dict['b_mat_train'], input_dict['b_mat_test']
        D, W = input_dict['D'], input_dict['W']
        # lambd = input_dict['lambd']
        # ista_step = input_dict['ista_step']
        self.m, self.n = D.shape
        self.output_size = self.n

        self.k_steps_train_fn = partial(k_steps_train_alista, D=D, W=W,
                                        jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_alista, D=D, W=W,
                                       jit=self.jit)
        self.out_axes_length = 5

    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = self.supervised and diff_required
        loss_method = self.loss_method

        def predict(params, input, q, iters, z_star, key, factor):
            z0 = jnp.zeros(z_star.size)

            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn

            # w_key = random.split(key)
            w_key = random.PRNGKey(key)
            perturb = random.normal(w_key, (self.train_unrolls, 2))
            # return scale * random.normal(w_key, (n, m))
            if self.deterministic:
                stochastic_params = params[0]
            else:
                stochastic_params = params[0] + jnp.sqrt(jnp.exp(params[1])) * perturb

            if diff_required:
                z_final, iter_losses = train_fn(k=iters,
                                                    z0=z0,
                                                    q=q,
                                                    params=stochastic_params,
                                                    supervised=supervised,
                                                    z_star=z_star)
            else:
                eval_out = eval_fn(k=iters,
                                    z0=z0,
                                    q=q,
                                    params=stochastic_params,
                                    supervised=supervised,
                                    z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                angles = None

            loss = self.final_loss(loss_method, z_final, iter_losses, supervised, z0, z_star)

            penalty_loss = calculate_total_penalty(self.N_train, params, self.b, self.c, self.delta)
            loss = loss + self.penalty_coeff * penalty_loss

            if diff_required:
                return loss
            else:
                return_out = (loss, iter_losses, z_all_plus_1, angles) + eval_out[3:]
                return return_out
        loss_fn = self.predict_2_loss(predict, diff_required)
        return loss_fn
