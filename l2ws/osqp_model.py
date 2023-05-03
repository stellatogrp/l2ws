from l2ws.l2ws_model import L2WSmodel
import time
import jax.numpy as jnp
from l2ws.algo_steps import k_steps_eval_osqp, k_steps_train_osqp
from functools import partial
from jax import vmap, jit


class OSQPmodel(L2WSmodel):
    def __init__(self, input_dict):
        super(OSQPmodel, self).__init__(input_dict)

    def initialize_algo(self, input_dict):
        A = input_dict['A']
        factor = input_dict['factor']
        m, n = A.shape
        self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']
        rho = input_dict.get('rho', 1)
        sigma = input_dict.get('sigma', 1)

        self.output_size = n + m

        self.k_steps_train_fn = partial(k_steps_train_osqp, factor=factor, A=A, rho=rho, sigma=sigma, jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_osqp, factor=factor, A=A, rho=rho, sigma=sigma, jit=self.jit)
