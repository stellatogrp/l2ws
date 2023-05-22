from l2ws.l2ws_model import L2WSmodel
import time
import jax.numpy as jnp
from l2ws.algo_steps import k_steps_eval_osqp, k_steps_train_osqp
from functools import partial
from jax import vmap, jit
import osqp
import numpy as np
from scipy.sparse import csc_matrix


class OSQPmodel(L2WSmodel):
    def __init__(self, input_dict):
        super(OSQPmodel, self).__init__(input_dict)


    def initialize_algo(self, input_dict):
        self.A = input_dict['A']
        self.P = input_dict.get('P', None)
        factor = input_dict['factor']
        self.m, self.n = self.A.shape
        self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']
        # rho = input_dict.get('rho', 1)
        rho = input_dict['rho']

        sigma = input_dict.get('sigma', 1)

        self.output_size = self.n + self.m

        self.k_steps_train_fn = partial(k_steps_train_osqp, factor=factor, A=self.A, rho=rho, sigma=sigma, jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_osqp, factor=factor, P=self.P, A=self.A, rho=rho, sigma=sigma, jit=self.jit)
        self.out_axes_length = 6


    def solve_c(self, z0_mat, q_mat, rel_tol, abs_tol, max_iter=10000):
        # assume M doesn't change across problems
        # static problem data
        m, n = self.m, self.n
        P, A = self.P, self.A

        # set the solver
        # b_zeros, c_zeros = np.zeros(m), np.zeros(n)

        osqp_solver = osqp.OSQP()
        P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))

        # q = q_mat[0, :]
        c, l, u = np.zeros(n), np.zeros(m), np.zeros(m)
        osqp_solver.setup(P=P_sparse, q=c, A=A_sparse, l=l, u=u, alpha=1, rho=1, sigma=1, polish=False,
                        adaptive_rho=False, scaling=0, max_iter=max_iter, verbose=True, eps_abs=abs_tol, eps_rel=rel_tol)

        num = z0_mat.shape[0]
        solve_times = np.zeros(num)
        solve_iters = np.zeros(num)
        x_sols = jnp.zeros((num, n))
        y_sols = jnp.zeros((num, m))
        for i in range(num):
            # set c, l, u
            c, l, u = q_mat[i, :n], q_mat[i, n:n + m], q_mat[i, n + m:]
            osqp_solver.update(q=np.array(c))
            osqp_solver.update(l=np.array(l), u=np.array(u))

            # set the warm start
            # x, y, s = self.get_xys_from_z(z0_mat[i, :])
            x_ws, y_ws = np.array(z0_mat[i, :n]), np.array(z0_mat[i, n:n + m])

            # fix warm start
            osqp_solver.warm_start(x=x_ws, y=y_ws)

            # solve
            results = osqp_solver.solve()
            # sol = solver.solve(warm_start=True, x=np.array(x), y=np.array(y), s=np.array(s))

            # set the solve time in seconds
            solve_times[i] = results.info.solve_time
            solve_iters[i] = results.info.iter

            # set the results
            x_sols = x_sols.at[i, :].set(results.x)
            y_sols = y_sols.at[i, :].set(results.y)

        return solve_times, solve_iters, x_sols, y_sols
