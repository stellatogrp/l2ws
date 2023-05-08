from l2ws.l2ws_model import L2WSmodel
import jax.numpy as jnp
from l2ws.algo_steps import k_steps_eval_ista, k_steps_train_ista, k_steps_eval_fista, k_steps_train_fista
from functools import partial


class GDmodel(L2WSmodel):
    def __init__(self, input_dict):
        # self.fista = input_dict['algorithm'] == 'fista'
        super(GDmodel, self).__init__(input_dict)

    def initialize_algo(self, input_dict):
        self.q_mat_train, self.q_mat_test = input_dict['b_mat_train'], input_dict['b_mat_test']
        P = input_dict['P']
        # lambd = input_dict['lambd']
        gd_step = input_dict['gd_step']
        n = P.shape[0]
        self.output_size = n

        # if self.fista:
        #     self.k_steps_train_fn = partial(k_steps_train_fista, A=A, lambd=lambd, ista_step=ista_step, jit=self.jit)
        #     self.k_steps_eval_fn = partial(k_steps_eval_fista, A=A, lambd=lambd, ista_step=ista_step, jit=self.jit)
        # else:
        self.k_steps_train_fn = partial(k_steps_train_ista, P=P, gd_step=gd_step, jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_ista, P=P, gd_step=gd_step, jit=self.jit)
