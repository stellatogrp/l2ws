from functools import partial
import jax.scipy as jsp
import jax.numpy as jnp
import cvxpy as cp
from matplotlib import pyplot as plt
from jax import random
import jax
from l2ws.algo_steps import create_M, create_projection_fn, lin_sys_solve, fixed_point, \
    fixed_point_hsde, extract_sol
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

    iter_losses = jnp.zeros(iters)

    if hsde:
        vec_length = m + n + 1
        r = lin_sys_solve(algo_factor, q)
        fp = partial(fixed_point_hsde, homogeneous=True, r=r, factor=algo_factor, proj=proj)
    else:
        vec_length = m + n
        fp = partial(fixed_point, q=q, factor=algo_factor, proj=proj)

    z_all = jnp.zeros((iters, vec_length))
    u_tilde_all = jnp.zeros((iters, vec_length))
    u_all = jnp.zeros((iters, vec_length))
    v_all = jnp.zeros((iters, vec_length))

    def body_fn(i, val):
        z, iter_losses, z_all, u_all, u_tilde_all, v_all = val
        z_next, u, u_tilde, v = fp(z)
        diff = jnp.linalg.norm(z_next - z)
        iter_losses = iter_losses.at[i].set(diff)
        z_all = z_all.at[i, :].set(z_next)
        u_all = u_all.at[i, :].set(u)
        u_tilde_all = u_tilde_all.at[i, :].set(u_tilde)
        v_all = v_all.at[i, :].set(v)
        val = z_next, iter_losses, z_all, u_all, u_tilde_all, v_all
        return val

    if hsde:
        # first step: iteration 0
        # we set homogeneous = False for the first iteration
        #   to match the SCS code which has the global variable FEASIBLE_ITERS
        #   which is set to 1
        z_next, u, u_tilde, v = fixed_point_hsde(z, False, r, algo_factor, proj)
        z_all = z_all.at[0, :].set(z)
        u_all = u_all.at[0, :].set(u)
        u_tilde_all = u_tilde_all.at[0, :].set(u_tilde)
        v_all = v_all.at[0, :].set(v)
        iter_losses = iter_losses.at[0].set(jnp.linalg.norm(z_next - z))
        z = z_next

    # fori_loop for iterations start, ..., iters - 1
    # for the rest of the iterations, we set homogeneous = False
    #   which forces us to use the root_calc function
    start = 1 if hsde else 0
    init_val = z, iter_losses, z_all, u_all, u_tilde_all, v_all
    if jit:
        val = jax.lax.fori_loop(start, iters, body_fn, init_val)
    else:
        val = python_fori_loop(start, iters, body_fn, init_val)

    z, iter_losses, z_all, u_all, u_tilde_all, v_all = val

    u_final, v_final = u_all[-1, :], v_all[-1, :]

    # extract the primal and dual variables
    x, y, s = extract_sol(u_final, v_final, n, hsde)

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
