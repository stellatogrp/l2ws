from functools import partial

from l2ws.algo_steps import (
    k_steps_eval_ista,
    k_steps_train_ista,
)
from l2ws.l2ws_model import L2WSmodel


class ISTAmodel(L2WSmodel):
    def __init__(self, **kwargs):
        super(ISTAmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'ista'
        self.factors_required = False
        self.q_mat_train, self.q_mat_test = input_dict['b_mat_train'], input_dict['b_mat_test']
        A = input_dict['A']
        lambd = input_dict['lambd']
        ista_step = input_dict['ista_step']
        m, n = A.shape
        self.output_size = n

        self.k_steps_train_fn = partial(k_steps_train_ista, A=A, lambd=lambd, 
                                        ista_step=ista_step, jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_ista, A=A, lambd=lambd, 
                                       ista_step=ista_step, jit=self.jit)
        self.out_axes_length = 5
