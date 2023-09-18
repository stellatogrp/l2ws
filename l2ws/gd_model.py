from l2ws.l2ws_model import L2WSmodel
import jax.numpy as jnp
from l2ws.algo_steps import k_steps_eval_gd, k_steps_train_gd
from functools import partial


class GDmodel(L2WSmodel):
    def __init__(self, input_dict):
        # self.fista = input_dict['algorithm'] == 'fista'
        super(GDmodel, self).__init__(input_dict)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'gd'
        self.factors_required = False
        self.q_mat_train, self.q_mat_test = input_dict['c_mat_train'], input_dict['c_mat_test']
        P = input_dict['P']
        gd_step = input_dict['gd_step']
        n = P.shape[0]
        self.output_size = n

        self.k_steps_train_fn = partial(k_steps_train_gd, P=P, gd_step=gd_step, jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_gd, P=P, gd_step=gd_step, jit=self.jit)
        self.out_axes_length = 5
