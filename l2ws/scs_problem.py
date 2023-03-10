from functools import partial
import jax.scipy as jsp
import jax.numpy as jnp
import cvxpy as cp
from matplotlib import pyplot as plt
from jax import random, lax
import jax
from l2ws.algo_steps import create_M, create_projection_fn, lin_sys_solve, fixed_point, \
    fixed_point_hsde, extract_sol, k_steps_eval
from l2ws.utils.generic_utils import python_fori_loop


class SCSinstance(object):
    def __init__(self, prob, solver, manual_canon=False):
        self.manual_canon = manual_canon
        if manual_canon:
            # manual canonicalization
            data = prob
            self.P = data['P']
            self.A = data['A']
            self.b = data['b']
            self.c = data['c']
            self.scs_data = dict(P=self.P, A=self.A, b=self.b, c=self.c)
            # self.cones = data['cones']
            solver.update(b=self.b)
            solver.update(c=self.c)
            self.solver = solver

        else:
            # automatic canonicalization
            data = prob.get_problem_data(cp.SCS)[0]
            self.P = data['P']
            self.A = data['A']
            self.b = data['b']
            self.c = data['c']
            self.cones = dict(z=data['dims'].zero, l=data['dims'].nonneg)
            # self.cones = dict(data['dims'].zero, data['dims'].nonneg)
            self.prob = prob

        # we will use self.q for our DR-splitting

        self.q = jnp.concatenate([self.c, self.b])
        self.solve()

    def solve(self):
        if self.manual_canon:
            # solver = scs.SCS(self.scs_data, self.cones,
            #                  eps_abs=1e-4, eps_rel=1e-4)
            # solver = scs.SCS(self.scs_data, self.cones,
            #                  eps_abs=1e-5, eps_rel=1e-5)
            # Solve!
            sol = self.solver.solve()
            self.x_star = jnp.array(sol['x'])
            self.y_star = jnp.array(sol['y'])
            self.s_star = jnp.array(sol['s'])
            self.solve_time = sol['info']['solve_time'] / 1000
        else:
            self.prob.solve(solver=cp.SCS, verbose=True)
            self.x_star = jnp.array(
                self.prob.solution.attr['solver_specific_stats']['x'])
            self.y_star = jnp.array(
                self.prob.solution.attr['solver_specific_stats']['y'])
            self.s_star = jnp.array(
                self.prob.solution.attr['solver_specific_stats']['s'])


def scs_jax(data, hsde=True, iters=5000, jit=True, plot=False):
    P, A = data['P'], data['A']
    c, b = data['c'], data['b']
    cones = data['cones']

    m, n = A.shape

    M = create_M(P, A)
    algo_factor = jsp.linalg.lu_factor(M + jnp.eye(m+n))
    q = jnp.concatenate([c, b])

    proj = create_projection_fn(cones, n)

    key = random.PRNGKey(0)
    if 'x' in data.keys() and 'y' in data.keys() and 's' in data.keys():
        # warm start with z = (x, y + s) or
        # z = (x, y + s, 1) with the hsde
        z = jnp.concatenate([data['x'], data['y'] + data['s']])
        if hsde:
            # we pick eta = 1 for feasibility of warm-start
            z = jnp.concatenate([z, jnp.array([1])])
    else:
        if hsde:
            mu = 1 * random.normal(key, (m + n,))
            z = jnp.concatenate([mu, jnp.array([1])])
        else:
            z = 1 * random.normal(key, (m + n,))

    if hsde:
        q_r = lin_sys_solve(algo_factor, q)
    else:
        q_r = q

    eval_out = k_steps_eval(iters, z, q_r, algo_factor, proj, P, A, c, b, jit, hsde)
    z_final, iter_losses, primal_residuals, dual_residuals, z_all_plus_1, u_all, v_all = eval_out

    u_final, v_final = u_all[-1, :], v_all[-1, :]

    # extract the primal and dual variables
    x, y, s = extract_sol(u_final, v_final, n, hsde)

    # import pdb
    # pdb.set_trace()

    if plot:
        plt.plot(iter_losses, label='fixed point residuals')
        plt.yscale('log')
        plt.legend()
        plt.show()

    # populate the sol dictionary
    sol = {}
    sol['fixed_point_residuals'] = iter_losses
    sol['x'], sol['y'], sol['s'] = x, y, s
    return sol


def ruiz_equilibrate(M, num_passes=20):
    p, p_ = M.shape
    D, E = jnp.eye(p), jnp.eye(p)
    val = M, E, D

    def body(i, val):
        M, E, D = val
        drinv = 1 / jnp.sqrt(jnp.linalg.norm(M, jnp.inf, axis=1))
        dcinv = 1 / jnp.sqrt(jnp.linalg.norm(M, jnp.inf, axis=0))
        D = jnp.multiply(D, drinv)
        E = jnp.multiply(E, dcinv)
        M = jnp.multiply(M, dcinv)
        M = jnp.multiply(drinv[:, None], M)
        val = M, E, D
        return val
    val = jax.lax.fori_loop(0, num_passes, body, val)
    M, E, D = val

    # for i in range(num_passes):
    #     drinv = 1 / jnp.sqrt(jnp.linalg.norm(M, jnp.inf, axis=1))
    #     dcinv = 1 / jnp.sqrt(jnp.linalg.norm(M, jnp.inf, axis=0))
    #     D = jnp.multiply(D, drinv)
    #     E = jnp.multiply(E, dcinv)
    #     M = jnp.multiply(M, dcinv)
    #     M = jnp.multiply(drinv[:, None], M)
    return M, E, D
