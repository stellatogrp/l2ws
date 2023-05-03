import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from scipy.linalg import solve_discrete_are
from examples.osc_mass import static_canon_osqp
import cvxpy as cp


def generate_static_prob_data(nx, nu, seed):
    np.random.seed(seed)
    x_bar = 1 + np.random.rand(nx)
    u_bar = .1 * np.random.rand(nu)

    dA = .1 * np.random.normal(size=(nx, nx))
    orig_Ad = np.eye(nx) + dA

    # normalize Ad so eigenvalues are less than 1
    orig_evals, evecs = np.linalg.eig(orig_Ad)
    # import pdb
    # pdb.set_trace()
    max_norm = np.max(np.abs(orig_evals))
    if max_norm >= 1:
        Ad = orig_Ad / max_norm
    else:
        Ad = orig_Ad

    # evals = np.clip(orig_evals, a_min=-np.inf, a_max=1)
    # Ad = evecs @ np.diag(evals) @ evecs.T

    # new_evals, new_evecs = np.linalg.eigh(orig_Ad)

    Bd = np.random.normal(size=(nx, nu))

    # create cost matrices
    R = .1 * np.eye(nu)

    q_vec = np.random.rand(nx) * 10
    p = 0.7 #1.0
    q_vec_mask = np.random.choice([0, 1], size=(nx), p=[1-p, p], replace=True)
    q_vec = np.multiply(q_vec, q_vec_mask)
    Q = np.diag(q_vec)

    QT = solve_discrete_are(Ad, Bd, Q, R)
    # QT = np.eye(nx)

    return Ad, Bd, Q, QT, R, x_bar, u_bar


def multiple_random_mpc_osqp(N, T=10, x_init_box=2, nx=20, nu=10,
                             sigma=1, rho=1,
                             Ad=None,
                             Bd=None,
                             seed=42):
    # np.random.seed(seed)
    # Ad = np.random.normal(size=(nx, nx))
    # Bd = np.random.normal(size=(nx, nu))
    Ad, Bd, Q, QT, R, x_bar, u_bar = generate_static_prob_data(nx, nu, seed)
    static_dict = static_canon_osqp(T, nx, nu, x_bar, u_bar, Q,
                                    QT, R, Ad=Ad, Bd=Bd)
    P, A = static_dict['P'], static_dict['A']
    c, l, u = static_dict['c'], static_dict['l'], static_dict['u']
    m, n = A.shape
    Ad = static_dict['A_dynamics']
    cones = static_dict['cones']

    q_mat = jnp.zeros((N, n + 2 * m))
    q_mat = q_mat.at[:, :n].set(c)

    # factor
    M = P + sigma * jnp.eye(n) + rho * A.T @ A
    factor = jsp.linalg.lu_factor(M)

    # x_init is theta
    # x_init_mat = jnp.array(x_init_box * (2 * np.random.rand(N, nx) - 1))
    x_init_mat = .3 * jnp.array(x_bar * (2 * np.random.rand(N, nx) - 1))

    for i in range(N):
        # generate new rhs of first block constraint
        l = l.at[:nx].set(Ad @ x_init_mat[i, :])
        u = u.at[:nx].set(Ad @ x_init_mat[i, :])

        q_osqp = jnp.concatenate([c, l, u])
        q_mat = q_mat.at[i, :].set(q_osqp)
    theta_mat = x_init_mat
    return factor, P, A, q_mat, theta_mat

def solve_many_probs_cvxpy(P, A, q_mat):
    """
    solves many QPs where each problem has a different b vector
    """
    P = cp.atoms.affine.wraps.psd_wrap(P)
    m, n = A.shape
    N = q_mat.shape[0]
    x, w = cp.Variable(n), cp.Variable(m)
    c_param, l_param, u_param = cp.Parameter(n), cp.Parameter(m), cp.Parameter(m)
    constraints = [A @ x == w, l_param <= w, w <= u_param]
    # import pdb
    # pdb.set_trace()
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, P) + c_param @ x), constraints)
    # prob = cp.Problem(cp.Minimize(.5 * cp.sum_squares(np.array(A) @ z - b_param) + lambd * cp.tv(z)))
    z_stars = jnp.zeros((N, m + n))
    objvals = jnp.zeros((N))
    for i in range(N):
        c_param.value = np.array(q_mat[i, :n])
        l_param.value = np.array(q_mat[i, n:n + m])
        u_param.value = np.array(q_mat[i, n + m:])
        prob.solve(verbose=False)
        objvals = objvals.at[i].set(prob.value)

        # import pdb
        # pdb.set_trace()
        x_star = jnp.array(x.value)
        w_star = jnp.array(w.value)
        y_star = jnp.array(constraints[0].dual_value)
        # z_star = jnp.concatenate([x_star, w_star, y_star])
        z_star = jnp.concatenate([x_star, w_star])
        z_stars = z_stars.at[i, :].set(z_star)
    print('finished solving cvxpy problems')
    return z_stars, objvals
