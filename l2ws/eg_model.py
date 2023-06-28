from l2ws.l2ws_model import L2WSmodel
import jax.numpy as jnp
from l2ws.algo_steps import k_steps_eval_extragrad, k_steps_train_extragrad
from functools import partial


class EGmodel(L2WSmodel):
    def __init__(self, input_dict):
        super(EGmodel, self).__init__(input_dict)

    def initialize_algo(self, input_dict):
        self.algo = 'extragradient'
        self.factors_required = False
        self.factor_static = None
        self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']
        Q, R = input_dict['Q'], input_dict['R']
        eg_step = input_dict['eg_step']
        m = R.shape[0]
        n = Q.shape[0]
        self.output_size = m + n

        self.k_steps_train_fn = partial(k_steps_train_extragrad, Q=Q, R=R, eg_step=eg_step, jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_extragrad, Q=Q, R=R, eg_step=eg_step, jit=self.jit)
        self.out_axes_length = 5
