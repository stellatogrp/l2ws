from functools import partial

from l2ws.algo_steps import k_steps_eval_gd, k_steps_train_gd
from l2ws.l2ws_model import L2WSmodel


class GDmodel(L2WSmodel):
    def __init__(self, **kwargs):
        super(GDmodel, self).__init__(**kwargs)

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
