from functools import partial
import nntplib
from tabnanny import verbose
import jax.scipy as jsp
import jax.numpy as jnp
import cvxpy as cp
import pdb
import time
from matplotlib import pyplot as plt
import scipy
import numpy as np
from scipy.sparse import coo_matrix, bmat
import scs
from jax import jit, random
import functools
import jax


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


def scs_jax(data, iters=5000):
    ##### QUICK TEST
    # A = jnp.array([[1, 2],
    #     [-1, 3]])
    # def linoperator(v):
    #     return A@v
    # rhs = jnp.ones(2)
    # out_cg, info = jsp.sparse.linalg.cg(linoperator, rhs, tol=1e-09)
    # out_lin_solve = jnp.linalg.solve(A, rhs)
    # pdb.set_trace()
    #####

    P, A = data['P'], data['A']
    c, b = data['c'], data['b']
    cones = data['cones']
    zero_cone_int = cones['z']
    print('zero_cone_int', zero_cone_int)
    m, n = A.shape

    # create the matrix M
    M = jnp.zeros((n + m, n + m))
    M = M.at[:n, :n].set(P)
    M = M.at[:n, n:].set(A.T)
    M = M.at[n:, :n].set(-A)
    algo_factor = jsp.linalg.lu_factor(M + jnp.eye(m+n))
    q = jnp.concatenate([c, b])

    @jit
    def proj(input):
        proj = jnp.clip(input[n+zero_cone_int:], a_min=0)
        return jnp.concatenate([input[:n+zero_cone_int], proj])

    M_plus_I = M + jnp.eye(n + m)
    mat_inv = jnp.linalg.inv(M_plus_I)

    @jit
    def lin_sys_solve(rhs):
        return mat_inv @ rhs
        # return jsp.linalg.lu_solve(algo_factor, rhs)

    # lin_op = lambda x: A@x
    # def lin_op(vec):
    #     return P@vec + 3*vec
        # return M_plus_I @ vec
        # x, y = vec[:n], vec[n:]
        # Px = P @ x
        # split = int(m/2)
        # top = Px + y[:split] - y[split:]
        # bottom = jnp.concatenate([-x, x]) + y
        # out = jnp.concatenate([top, bottom])
        # return out

    # @jit
    # split = int(m/2)
    # def lin_sys_solve(rhs):
        # rx = rhs[:n]
        # ry = rhs[n:]
        # rhs_x = rx + ry[:split] - ry[split:]
        # x_sol, info = jsp.sparse.linalg.cg(lin_op, rhs_x, tol=1e-09)
        # sol = jsp.sparse.linalg.cg(M_plus_I, rhs, tol=1e-09)[0]
        # Ax = jnp.concatenate([x_sol, -x_sol])
        # y_sol = Ax - rhs[n:]
        # sol = jnp.concatenate([x_sol, y_sol])
        # pdb.set_trace()
        # return sol
        # return jsp.linalg.lu_solve(algo_factor, rhs)

    key = random.PRNGKey(0)
    if 'x' in data.keys():
        xy = jnp.concatenate([data['x'], data['y']])
        z = M @ xy + xy + q
    else:
        z = 0*random.normal(key, (m + n,))

    iter_losses = jnp.zeros(iters)
    primal_residuals = jnp.zeros(iters)
    dual_residuals = jnp.zeros(iters)
    duality_gaps = jnp.zeros(iters)
    all_x_primals = jnp.zeros((iters, n))

    for i in range(iters):
        # print('iter', i)
        z_next, x, y = fixed_point(z, q, lin_sys_solve, proj)
        diff = jnp.linalg.norm(z_next - z)
        iter_losses = iter_losses.at[i].set(diff)
        # print('x', x[:5])

        v = y + z - 2*x

        # get (x, y, s) - soln to original prob
        xp = y[:n]
        yd = y[n:]
        sp = v[n:]

        # get the residuals
        pr = jnp.linalg.norm(A @ xp + sp - b, ord=np.inf) #, ord='inf')
        dr = jnp.linalg.norm(A.T @ yd + P @ xp + c, ord=np.inf) #, ord='inf')
        dg = jax.lax.abs(xp @ P @ xp + c @ xp + b @ yd)

        if i % 10 == 0:
            print(f"iter {i}: loss: {diff}")
            print(f"iter {i}: primal residual: {pr}")
            print(f"iter {i}: dual residual: {dr}")
            print(f"iter {i}: duality gap: {dg}")
            print('x', xp[:5])
            print('y', yd[:5])
            print('z', z[-5:])

        # store the residuals
        primal_residuals = primal_residuals.at[i].set(pr)
        dual_residuals = dual_residuals.at[i].set(dr)
        duality_gaps = duality_gaps.at[i].set(dg)
        z = z_next
        all_x_primals = all_x_primals.at[i, :].set(x[:n])

    plt.plot(primal_residuals[5:], label='primal residuals')
    plt.plot(dual_residuals[5:], label='dual residuals')
    plt.plot(duality_gaps[5:], label='duality gaps')
    plt.plot(iter_losses[5:], label='fixed point residuals')
    plt.yscale('log')
    plt.legend()
    plt.show()

    # x_sol = x[:n]
    # y_sol = x[n:]
    # s_sol = v[n:]

    pdb.set_trace()
    return xp, yd, sp


# @functools.partial(jit, static_argnums=(2, 3,))
def fixed_point(z_init, q, lin_sys_solve, proj):
    x = lin_sys_solve(z_init - q)
    # pdb.set_trace()
    y_tilde = (2*x - z_init)
    y = proj(y_tilde)
    
    z = z_init +1.0*( y - x)
    return z, x, y


def ruiz_equilibrate(M, num_passes=20):
    p, p_ = M.shape
    D, E = np.eye(p), np.eye(p)
    for i in range(num_passes):
        Dr = np.diag(np.sqrt(np.linalg.norm(M, np.inf, axis=1)))
        Dc = np.diag(np.sqrt(np.linalg.norm(M, np.inf, axis=0)))
        Drinv = np.linalg.inv(Dr)
        Dcinv = np.linalg.inv(Dc)
        D = D @ Drinv
        E = E @ Dcinv
        M = Drinv @ M @ Dcinv
    return M
