from functools import partial
import jax.scipy as jsp
import jax.numpy as jnp
import cvxpy as cp
from matplotlib import pyplot as plt
from jax import random
import jax
from l2ws.algo_steps import create_M, create_projection_fn, lin_sys_solve, fixed_point, \
    fixed_point_hsde


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


def scs_jax(data, hsde=True, iters=5000, plot=False):
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
        # warm start with u = (x, y)
        z = jnp.concatenate([data['x'], data['y'] + data['s']])
        if hsde:
            # mu = M @ u + u + q

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
        z_all = jnp.zeros((iters, m + n + 1))
        u_tilde_all = jnp.zeros((iters, m + n + 1))
        u_all = jnp.zeros((iters, m + n + 1))
        p_all = jnp.zeros((iters, m + n))
        r = lin_sys_solve(algo_factor, q)
        fp = partial(fixed_point_hsde, root_plus_calc=True, r=r, factor=algo_factor, proj=proj)
    else:
        z_all = jnp.zeros((iters, m + n))
        u_tilde_all = jnp.zeros((iters, m + n))
        u_all = jnp.zeros((iters, m + n))
        fp = partial(fixed_point, q=q, factor=algo_factor, proj=proj)
    print('z0', z)

    def body_fn(i, val):
        z, iter_losses, z_all, u_all, u_tilde_all, p_all = val
        # z = z_init / jnp.linalg.norm(z_init)
        z_next, u, u_tilde, v, p = fp(z)
        diff = jnp.linalg.norm(z_next - z)
        iter_losses = iter_losses.at[i].set(diff)
        z_all = z_all.at[i, :].set(z_next)
        u_all = u_all.at[i, :].set(u)
        u_tilde_all = u_tilde_all.at[i, :].set(u_tilde)
        p_all = p_all.at[i, :].set(p)
        val = z_next, iter_losses, z_all, u_all, u_tilde_all, p_all
        return val
    # import pdb
    # pdb.set_trace()

    # first step
    z, u, u_tilde, v, p = fixed_point_hsde(z, False, r, algo_factor, proj)
    z_all = z_all.at[0, :].set(z)
    u_all = u_all.at[0, :].set(u)
    u_tilde_all = u_tilde_all.at[0, :].set(u_tilde)
    print('u', u)
    print('u_tilde', u_tilde)
    print('z', z)

    init_val = z, iter_losses, z_all, u_all, u_tilde_all, p_all

    # jax loop
    # val = jax.lax.fori_loop(1, iters, body_fn, init_val)

    # non jit loop
    def fori_loop(lower, upper, body_fun, init_val):
        val = init_val
        for i in range(lower, upper):
            val = body_fun(i, val)
        return val
    val = fori_loop(1, iters, body_fn, init_val)

    z, iter_losses, z_all, u_all, u_tilde_all, p_all = val
    print('z_all', z_all)
    print('u_tilde_all', u_tilde_all)
    print('u_all', u_all)
    print('p_all', p_all)

    # one more fixed point iteration
    z_next, u, v = fp(z)

    # extract the primal and dual variables
    if hsde:
        tao = u[-1]
        x, y, s = u[:n] / tao, u[n:-1] / tao, v[n:] / tao
    else:
        x, y, s = u[:n], u[n:], v[n:]

    if plot:
        plt.plot(iter_losses, label='fixed point residuals')
        plt.yscale('log')
        plt.legend()
        plt.show()
    sol = {}
    sol['fp_residuals'] = iter_losses
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
