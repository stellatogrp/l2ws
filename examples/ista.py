import jax.numpy as jnp
import numpy as np
import cvxpy as cp

def generate_b_mat(A, N, p=.1):
    m, n = A.shape
    b_mat = jnp.zeros((N, m))
    x_star_mask = np.random.binomial(1, p, size=(N, n))
    x_stars_dense = jnp.array(np.random.normal(size=(N, n)))
    x_stars = jnp.multiply(x_star_mask, x_stars_dense)
    for i in range(N):
        b = A @ x_stars[i, :]
        b_mat = b_mat.at[i, :].set(b)
    return b_mat

def eval_ista_obj(z, A, b, lambd):
    return .5 * jnp.linalg.norm(A @ z - b) ** 2 + lambd * jnp.linalg.norm(z, ord=1)


def obj_diff(obj, true_obj):
    return (obj - true_obj)


def sol_2_obj_diff(z, b, true_obj, A, lambd):
    obj = eval_ista_obj(z, A, b, lambd)
    return obj_diff(obj, true_obj)

def solve_many_probs_cvxpy(A, b_mat, lambd):
    """
    solves many lasso problems where each problem has a different b vector
    """
    m, n = A.shape
    N = b_mat.shape[0]
    z, b_param = cp.Variable(n), cp.Parameter(m)
    prob = cp.Problem(cp.Minimize(.5 * cp.sum_squares(np.array(A) @ z - b_param) + lambd * cp.norm(z, p=1)))
    z_stars = jnp.zeros((N, n))
    objvals = jnp.zeros((N))
    for i in range(N):
        b_param.value = np.array(b_mat[i, :])
        prob.solve()
        objvals = objvals.at[i].set(prob.value)
        z_stars = z_stars.at[i, :].set(jnp.array(z.value))
    print('finished solving cvxpy problems')
    return z_stars, objvals
