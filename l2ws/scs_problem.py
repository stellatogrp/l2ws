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
from jax import lax, vmap
from utils.generic_utils import vec_symm, unvec_symm


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
    # QUICK TEST
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
    nonneg_cone_int = cones['l']
    # num_soc = len(cones['q'])
    # soc_total = sum(cones['q'])
    print('zero_cone_int', zero_cone_int)
    m, n = A.shape
    # pdb.set_trace()

    # create the matrix M
    M = jnp.zeros((n + m, n + m))
    M = M.at[:n, :n].set(P)
    M = M.at[:n, n:].set(A.T)
    M = M.at[n:, :n].set(-A)
    algo_factor = jsp.linalg.lu_factor(M + jnp.eye(m+n))
    q = jnp.concatenate([c, b])

    soc = 'q' in cones.keys() and len(cones['q']) > 0
    sdp_ = 's' in cones.keys() and len(cones['s']) > 0

    if soc:
        num_soc = len(cones['q'])
        soc_total = sum(cones['q'])
        soc_cones_array = np.array(cones['q'])
        soc_size = soc_cones_array[0]
        soc_proj_single_batch = vmap(soc_proj_single, in_axes=(0), out_axes=(0))
    else:
        soc_total = 0
    if sdp_:
        num_sdp = len(cones['s'])
        sdp_total = sum(cones['s'])
        sdp_cones_array = np.array(cones['s'])
        sdp_size = int(sdp_cones_array[0] * (sdp_cones_array[0]+1) / 2)
        sdp_proj_single_dim = partial(sdp_proj_single, dim=sdp_cones_array[0])
        sdp_proj_single_batch = vmap(sdp_proj_single_dim, in_axes=(0), out_axes=(0))

    @jit
    def proj(input):
        nonneg = jnp.clip(input[n+zero_cone_int:n+zero_cone_int+nonneg_cone_int], a_min=0)
        projection = jnp.concatenate([input[:n+zero_cone_int], nonneg])
        if soc:
            socp = jnp.zeros(soc_total)
            soc_input = input[n+zero_cone_int+nonneg_cone_int:n +
                              zero_cone_int+nonneg_cone_int+soc_total]
            soc_input_reshaped = jnp.reshape(soc_input, (num_soc, soc_size))
            soc_out_reshaped = soc_proj_single_batch(soc_input_reshaped)
            socp = jnp.ravel(soc_out_reshaped)
            projection = jnp.concatenate([projection, socp])
        if sdp_:
            sdp = jnp.zeros(sdp_total)
            sdp_input = input[n + zero_cone_int+nonneg_cone_int+soc_total:]
            sdp_input_reshaped = jnp.reshape(sdp_input, (num_sdp, sdp_size))
            sdp_out_reshaped = sdp_proj_single_batch(sdp_input_reshaped)
            sdp = jnp.ravel(sdp_out_reshaped)
            projection = jnp.concatenate([projection, sdp])
        return projection

    # @jit
    # def proj(input):
    #     nonneg = jnp.clip(input[n+zero_cone_int:n+zero_cone_int+nonneg_cone_int], a_min=0)
    #     socp = jnp.zeros(soc_total)
    #     curr = 0 #zero_cone_int + nonneg_cone_int
    #     soc_input = input[n+zero_cone_int+nonneg_cone_int:]
    #     for i in range(num_soc):
    #         start = curr
    #         end = curr + cones['q'][i]
    #         # curr_soc_proj = soc_projection(input[start+1:end], input[start])
    #         curr_soc_proj = soc_projection(soc_input[start+1:end], soc_input[start])
    #         soc_concat = jnp.append(curr_soc_proj[1], curr_soc_proj[0])
    #         # curr_socp = start - (zero_cone_int + nonneg_cone_int)
    #         # end_socp = end - (zero_cone_int + nonneg_cone_int)
    #         # socp = socp.at[curr_socp:end_socp].set(soc_concat)
    #         socp = socp.at[curr:end].set(soc_concat)
    #         curr = end
    #         # pdb.set_trace()

    #     return jnp.concatenate([input[:n+zero_cone_int], nonneg, socp])

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
        z = 1*random.normal(key, (m + n,))

    # proj
    pw = proj(z)
    pdb.set_trace()

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
        pr = jnp.linalg.norm(A @ xp + sp - b, ord=np.inf)  # , ord='inf')
        dr = jnp.linalg.norm(A.T @ yd + P @ xp + c, ord=np.inf)  # , ord='inf')
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


def soc_proj_single(input):
    y, s = input[1:], input[0]
    pi_y, pi_s = soc_projection(y, s)
    return jnp.append(pi_s, pi_y)


def sdp_proj_single(x, dim):
    X = unvec_symm(x, dim)
    evals, evecs = jnp.linalg.eigh(X)
    evals_plus = jnp.clip(evals, 0, jnp.inf)
    X_proj = evecs @ jnp.diag(evals_plus) @ evecs.T
    x_proj = vec_symm(X_proj)
    return x_proj


def soc_projection(y, s):
    y_norm = jnp.linalg.norm(y)

    def case1_soc_proj(y, s):
        # case 1: y_norm >= |s|
        val = (s + y_norm) / (2 * y_norm)
        t = val * y_norm
        x = val * y
        return x, t

    def case2_soc_proj(y, s):
        # case 2: y_norm <= |s|
        # case 2a: s > 0
        # case 2b: s < 0
        def case2a(y, s):
            return y, s

        def case2b(y, s):
            return (0.0*jnp.zeros(2), 0.0)
        return lax.cond(s >= 0, case2a, case2b, y, s)
    return lax.cond(y_norm >= jnp.abs(s), case1_soc_proj, case2_soc_proj, y, s)


# @functools.partial(jit, static_argnums=(2, 3,))
def fixed_point(z_init, q, lin_sys_solve, proj):
    x = lin_sys_solve(z_init - q)
    # pdb.set_trace()
    y_tilde = (2*x - z_init)
    y = proj(y_tilde)

    z = z_init + 1.0*(y - x)
    return z, x, y


# def ruiz_equilibrate(M, num_passes=20):
#     p, p_ = M.shape
#     D, E = np.eye(p), np.eye(p)
#     for i in range(num_passes):
#         print('i', i)
#         # Dr = np.diag(np.sqrt(np.linalg.norm(M, np.inf, axis=1)))
#         # Dc = np.diag(np.sqrt(np.linalg.norm(M, np.inf, axis=0)))
#         # Drinv = np.linalg.inv(Dr)
#         # Dcinv = np.linalg.inv(Dc)
#         pdb.set_trace()
#         drinv = 1 / np.sqrt(np.linalg.norm(M, np.inf, axis=1))
#         dcinv = 1 / np.sqrt(np.linalg.norm(M, np.inf, axis=0))
#         # Drinv = np.diag(drinv)
#         # Dcinv = np.diag(dcinv)
#         # D = D @ Drinv
#         # E = E @ Dcinv
#         # D = np.multiply(D, Drinv)
#         # E = np.multiply(E, Dcinv)
#         D = np.multiply(D, drinv)
#         E = np.multiply(E, dcinv)
#         # M = Drinv @ M @ Dcinv
#         M = np.multiply(M, dcinv)
#         M = np.multiply(drinv[:,None], M)
#     return M, E, D

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
