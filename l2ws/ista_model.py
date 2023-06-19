from l2ws.l2ws_model import L2WSmodel
import jax.numpy as jnp
from l2ws.algo_steps import k_steps_eval_ista, k_steps_train_ista, k_steps_eval_fista, k_steps_train_fista
from functools import partial


class ISTAmodel(L2WSmodel):
    def __init__(self, input_dict):
        # self.fista = input_dict['algorithm'] == 'fista'
        super(ISTAmodel, self).__init__(input_dict)

    def initialize_algo(self, input_dict):
        self.algo = 'ista'
        self.factors_required = False
        self.q_mat_train, self.q_mat_test = input_dict['b_mat_train'], input_dict['b_mat_test']
        A = input_dict['A']
        lambd = input_dict['lambd']
        ista_step = input_dict['ista_step']
        m, n = A.shape
        self.output_size = n

        # if self.fista:
        #     self.k_steps_train_fn = partial(k_steps_train_fista, A=A, lambd=lambd, ista_step=ista_step, jit=self.jit)
        #     self.k_steps_eval_fn = partial(k_steps_eval_fista, A=A, lambd=lambd, ista_step=ista_step, jit=self.jit)
        # else:
        self.k_steps_train_fn = partial(k_steps_train_ista, A=A, lambd=lambd, ista_step=ista_step, jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_ista, A=A, lambd=lambd, ista_step=ista_step, jit=self.jit)
        self.out_axes_length = 5
