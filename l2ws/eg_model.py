from functools import partial

from l2ws.algo_steps import k_steps_eval_extragrad, k_steps_train_extragrad
from l2ws.l2ws_model import L2WSmodel


class EGmodel(L2WSmodel):
    def __init__(self, input_dict):
        super(EGmodel, self).__init__(input_dict)

    def initialize_algo(self, input_dict):
        self.m, self.n = input_dict['m'], input_dict['n']
        self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']
        self.algo = 'extragradient'
        self.factors_required = False
        self.factor_static = None

        eg_step = input_dict['eg_step']
        m, n = self.m, self.n

        # function
        proj_X, proj_Y = input_dict['proj_X'], input_dict['proj_Y']
        f = input_dict['f']

        self.output_size = m + n
        self.out_axes_length = 5
        # self.k_steps_train_fn = partial(k_steps_train_extragrad, Q=Q, R=R, eg_step=eg_step, jit=self.jit)
        self.k_steps_train_fn = partial(
            k_steps_train_extragrad, f=f, proj_X=proj_X, proj_Y=proj_Y, n=n, eg_step=eg_step, jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_extragrad,
                                       f=f, proj_X=proj_X, proj_Y=proj_Y, n=n, eg_step=eg_step, jit=self.jit)

        # old
        # self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']
        # m = R.shape[0]
        # n = Q.shape[0]
        # Q, R = input_dict['Q'], input_dict['R']
        # self.k_steps_train_fn = partial(k_steps_train_extragrad, Q=Q, R=R, eg_step=eg_step, jit=self.jit)
        # self.k_steps_eval_fn = partial(k_steps_eval_extragrad, Q=Q, R=R, eg_step=eg_step, jit=self.jit)
