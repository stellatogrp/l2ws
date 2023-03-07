import jax.numpy as jnp
from jax import lax, vmap, jit
from utils.generic_utils import vec_symm, unvec_symm
from functools import partial
import jax.scipy as jsp


def create_projection_fn(cones, n):
    """
    cones is a dict with keys
    z: zero cone
    l: non-negative cone
    q: second-order cone
    s: positive semidefinite cone

    n is the size of the variable x in the problem
    min 1/2 x^T P x + c^T x
        s.t. Ax + s = b
             s in K
    This function returns a projection Pi
    which is defined by
    Pi(w) = argmin_v ||w - v||_2^2
                s.t. v in C
    where
    C = {0}^n x K^*
    i.e. the cartesian product of the zero cone of length n and the dual
        cone of K
    For all of the cones we consider, the cones are self-dual
    """
    zero_cone, nonneg_cone = cones['z'], cones['l']
    soc = 'q' in cones.keys() and len(cones['q']) > 0
    sdp = 's' in cones.keys() and len(cones['s']) > 0
    if soc:
        soc_cones_array = jnp.array(cones['q'])

        # soc_proj_sizes, soc_num_proj are lists
        # need to convert to list so that the item is not a traced object
        soc_proj_sizes, soc_num_proj = count_num_repeated_elements(soc_cones_array)
    else:
        soc_proj_sizes, soc_num_proj = [], []
    if sdp:
        sdp_cones_array = jnp.array(cones['s'])

        # soc_proj_sizes, soc_num_proj are lists
        # need to convert to list so that the item is not a traced object
        sdp_row_sizes, sdp_num_proj = count_num_repeated_elements(sdp_cones_array)
        sdp_vector_sizes = [int(row_size * (row_size + 1) / 2) for row_size in sdp_row_sizes]
    else:
        sdp_row_sizes, sdp_vector_sizes, sdp_num_proj = [], [], []

    projection = partial(proj,
                         n=n,
                         zero_cone_int=int(zero_cone),
                         nonneg_cone_int=int(nonneg_cone),
                         soc_proj_sizes=soc_proj_sizes,
                         soc_num_proj=soc_num_proj,
                         sdp_row_sizes=sdp_row_sizes,
                         sdp_vector_sizes=sdp_vector_sizes,
                         sdp_num_proj=sdp_num_proj,
                         )
    return jit(projection)


def get_psd_sizes(cones):
    """
    returns a list with the size of the psd projections
    """
    sdp = 's' in cones.keys() and len(cones['s']) > 0
    if sdp:
        sdp_cones_array = jnp.array(cones['s'])
        psd_sizes = sdp_cones_array.tolist()
    else:
        psd_sizes = [0]
    return psd_sizes


def root_plus(mu, eta, p, r):
    """
    mu, p, r are vectors each with size (m + n)
    eta is a scalar

    A step that solves the linear system
    (I + M)z + q tao = mu^k
    tao^2 - tao(eta^k + z^Tq) - z^T M z = 0
    where z in reals^d and tao > 0

    Since M is monotone, z^T M z >= 0
    Quadratic equation will have one non-negative root and one non-positive root

    solve by substituting z = p^k - r tao
        where r = (I + M)^{-1} q
        and p^k = (I + M)^{-1} mu^k

    the result is a closed-form solution involving the quadratic formula
        we take the positive root
    """
    a = 1 + r @ r
    b = r @ mu - 2 * r @ p - eta
    c = p @ (p - mu)
    return (-b + jnp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def fixed_point(z_init, q, factor, proj):
    """
    implements 1 iteration of algorithm 1 in https://arxiv.org/pdf/2212.08260.pdf
    """
    u_tilde = lin_sys_solve(factor, z_init - q)
    u_temp = 2*u_tilde - z_init
    u = proj(u_temp)
    v = u + z_init - 2*u_tilde
    z = z_init + u - u_tilde
    return z, u, v


def fixed_point_hsde(z_init, r, factor, proj):
    """
    implements 1 iteration of algorithm 5.1 in https://arxiv.org/pdf/2004.02177.pdf

    the names of the variables are a bit different compared with that paper

    we have
    u_tilde = (w_tilde, tao_tilde)
    u = (w, tao)
    z = (mu, eta)

    they have
    u_tilde = (z_tilde, tao_tilde)
    u = (z, tao)
    w = (mu, eta)

    tao_tilde, tao, eta are all scalars
    w_tilde, w, mu all have size (m + n)

    r = (I + M)^{-1} q
    requires the inital eta > 0
    """
    # z = (mu, eta)
    mu, eta = z_init[:-1], z_init[-1]

    # u_tilde, tao_tilde update
    p = lin_sys_solve(factor, mu)
    tao_tilde = root_plus(mu, eta, p, r)
    w_tilde = p - r * tao_tilde

    # u, tao update
    w_temp = 2 * w_tilde - mu
    w = proj(w_temp)
    tao = jnp.clip(2 * tao_tilde - eta, a_min=0)

    # for s extraction - not needed for algorithm
    v = w + mu - 2 * w_tilde

    # mu, eta update
    mu = mu + w - w_tilde
    eta = eta + tao - tao_tilde

    # concatenate for z, u
    z = jnp.concatenate([mu, jnp.array([eta])])
    u = jnp.concatenate([w, jnp.array([tao])])

    # z and u have size (m + n + 1)
    # v has shape (m + n)
    return z, u, v


def create_M(P, A):
    """
    create the matrix M in jax
    M = [ P   A.T
         -A   0  ]
    """
    m, n = A.shape
    M = jnp.zeros((n + m, n + m))
    M = M.at[:n, :n].set(P)
    M = M.at[:n, n:].set(A.T)
    M = M.at[n:, :n].set(-A)
    return M


def lin_sys_solve(factor, b):
    """
    solves the linear system
    Ax = b
    where factor is the lu factorization of A
    """
    return jsp.linalg.lu_solve(factor, b)


def proj(input, n, zero_cone_int, nonneg_cone_int, soc_proj_sizes, soc_num_proj, sdp_row_sizes,
         sdp_vector_sizes, sdp_num_proj):
    """
    projects the input onto a cone which is a cartesian product of the zero cone,
        non-negative orthant, many second order cones, and many positive semidefinite cones

    Assumes that the ordering is as follows
    zero, non-negative orthant, second order cone, psd cone
    ============================================================================
    SECOND ORDER CONE
    soc_proj_sizes: list of the sizes of the socp projections needed
    soc_num_proj: list of the number of socp projections needed for each size

    e.g. 50 socp projections of size 3 and 1 projection of size 100 would be
    soc_proj_sizes = [3, 100]
    soc_num_proj = [50, 1]
    ============================================================================
    PSD CONE
    sdp_proj_sizes: list of the sizes of the sdp projections needed
    sdp_vector_sizes: list of the sizes of the sdp projections needed
    soc_num_proj: list of the number of socp projections needed for each size

    e.g. 3 sdp projections of size 10, 10, and 100 would be
    sdp_proj_sizes = [10, 100]
    sdp_vector_sizes = [55, 5050]
    sdp_num_proj = [2, 1]
    """
    nonneg = jnp.clip(input[n + zero_cone_int: n + zero_cone_int + nonneg_cone_int], a_min=0)
    projection = jnp.concatenate([input[:n + zero_cone_int], nonneg])

    # soc setup
    num_soc_blocks = len(soc_proj_sizes)

    # avoiding doing inner product using jax so that we can jit
    soc_total = sum(i[0] * i[1] for i in zip(soc_proj_sizes, soc_num_proj))
    soc_bool = num_soc_blocks > 0

    # sdp setup
    num_sdp_blocks = len(sdp_row_sizes)
    sdp_total = sum(i[0] * i[1] for i in zip(sdp_vector_sizes, sdp_num_proj))
    sdp_bool = num_sdp_blocks > 0

    if soc_bool:
        socp = jnp.zeros(soc_total)
        soc_input = input[n+zero_cone_int+nonneg_cone_int:n +
                          zero_cone_int+nonneg_cone_int + soc_total]

        # iterate over the blocks
        start = 0
        for i in range(num_soc_blocks):
            # calculate the end point
            end = start + soc_proj_sizes[i] * soc_num_proj[i]

            # extract the right soc_input
            curr_soc_input = lax.dynamic_slice(
                soc_input, (start,), (soc_proj_sizes[i] * soc_num_proj[i],))

            # reshape so that we vmap all of the socp projections of the same size together
            curr_soc_input_reshaped = jnp.reshape(
                curr_soc_input, (soc_num_proj[i], soc_proj_sizes[i]))
            curr_soc_out_reshaped = soc_proj_single_batch(curr_soc_input_reshaped)
            curr_socp = jnp.ravel(curr_soc_out_reshaped)

            # place in the correct location in the socp vector
            socp = socp.at[start:end].set(curr_socp)

            # update the start point
            start = end

        projection = jnp.concatenate([projection, socp])
    if sdp_bool:
        sdp_proj = jnp.zeros(sdp_total)
        sdp_input = input[n + zero_cone_int + nonneg_cone_int + soc_total:]

        # iterate over the blocks
        start = 0
        for i in range(num_sdp_blocks):
            # calculate the end point
            end = start + sdp_vector_sizes[i] * sdp_num_proj[i]

            # extract the right sdp_input
            curr_sdp_input = lax.dynamic_slice(
                sdp_input, (start,), (sdp_vector_sizes[i] * sdp_num_proj[i],))

            # reshape so that we vmap all of the sdp projections of the same size together
            curr_sdp_input_reshaped = jnp.reshape(
                curr_sdp_input, (sdp_num_proj[i], sdp_vector_sizes[i]))
            curr_sdp_out_reshaped = sdp_proj_batch(curr_sdp_input_reshaped, sdp_row_sizes[i])
            curr_sdp = jnp.ravel(curr_sdp_out_reshaped)

            # place in the correct location in the sdp vector
            sdp_proj = sdp_proj.at[start:end].set(curr_sdp)

            # update the start point
            start = end

        projection = jnp.concatenate([projection, sdp_proj])
    return projection


def count_num_repeated_elements(vector):
    """
    given a vector, outputs the frequency in a row

    e.g. vector = [5, 5, 10, 10, 5]

    val_repeated = [5, 10, 5]
    num_repeated = [2, 2, 1]
    """
    m = jnp.r_[True, vector[:-1] != vector[1:], True]
    counts = jnp.diff(jnp.flatnonzero(m))
    unq = vector[m[:-1]]
    out = jnp.c_[unq, counts]

    val_repeated = out[:, 0].tolist()
    num_repeated = out[:, 1].tolist()
    return val_repeated, num_repeated


def soc_proj_single(input):
    """
    input is a single vector
        input = (s, y) where y is a vector and s is a scalar
    then we call soc_projection
    """
    # break into scalar and vector parts
    y, s = input[1:], input[0]

    # do the projection
    pi_y, pi_s = soc_projection(y, s)

    # stitch the pieces back together
    return jnp.append(pi_s, pi_y)


def sdp_proj_single(x, n):
    """
    x_proj = argmin_y ||y - x||_2^2
                s.t.   y is psd
    x is a vector with shape (n * (n + 1) / 2)

    we need to pass in n to jit this function
        we could extract dim from x.shape theoretically,
        but we cannot jit a function
        whose values depend on the size of inputs
    """
    # convert vector of size (n * (n + 1) / 2) to matrix of shape (n, n)
    X = unvec_symm(x, n)

    # do the eigendecomposition of X
    evals, evecs = jnp.linalg.eigh(X)

    # clip the negative eigenvalues
    evals_plus = jnp.clip(evals, 0, jnp.inf)

    # put the projection together with non-neg eigenvalues
    X_proj = evecs @ jnp.diag(evals_plus) @ evecs.T

    # vectorize the matrix
    x_proj = vec_symm(X_proj)
    return x_proj


def soc_projection(x, s):
    """
    returns the second order cone projection of (x, s)
    (y, t) = Pi_{K}(x, s)
    where K = {y, t | ||y||_2 <= t}

    the second order cone admits a closed form solution

    (y, t) = alpha (x, ||x||_2) if ||x|| >= |s|
             (x, s) if ||x|| <= |s|, s >= 0
             (0, 0) if ||x|| <= |s|, s <= 0

    where alpha = (s + ||x||_2) / (2 ||x||_2)

    case 1: ||x|| >= |s|
    case 2: ||x|| >= |s|
        case 2a: ||x|| >= |s|, s >= 0
        case 2b: ||x|| <= |s|, s <= 0

    """
    x_norm = jnp.linalg.norm(x)

    def case1_soc_proj(x, s):
        # case 1: y_norm >= |s|
        val = (s + x_norm) / (2 * x_norm)
        t = val * x_norm
        y = val * x
        return y, t

    def case2_soc_proj(x, s):
        # case 2: y_norm <= |s|
        # case 2a: s > 0
        def case2a(x, s):
            return x, s

        # case 2b: s < 0
        def case2b(x, s):
            return (0.0*jnp.zeros(x.size), 0.0)
        return lax.cond(s >= 0, case2a, case2b, x, s)
    return lax.cond(x_norm >= jnp.abs(s), case1_soc_proj, case2_soc_proj, x, s)


# provides vmapped versions of the projections for the soc and psd cones
soc_proj_single_batch = vmap(soc_proj_single, in_axes=(0), out_axes=(0))
sdp_proj_batch = vmap(sdp_proj_single, in_axes=(0, None), out_axes=(0))

"""
attempt to use jax.fori_loop for multiple soc projections of different sizes
not possible according to https://github.com/google/jax/issues/2962

def soc_body(i, val):
    socp, start = val

    # calculate the end point
    # end = start + soc_proj_sizes[i] * soc_num_proj[i]

    # extract the right soc_input
    # curr_soc_input = soc_input[start:end]
    curr_soc_input = lax.dynamic_slice(
        soc_input, (start,), (soc_proj_sizes[i] * soc_num_proj[i],))

    # reshape so that we vmap all of the socp projections of the same size together
    curr_soc_input_reshaped = jnp.reshape(
        curr_soc_input, (soc_num_proj[i], soc_proj_sizes[i]))
    curr_soc_out_reshaped = soc_proj_single_batch(curr_soc_input_reshaped)
    curr_socp = jnp.ravel(curr_soc_out_reshaped)

    # calculate the end point
    end = start + soc_proj_sizes[i] * soc_num_proj[i]

    # place in the correct location in the socp vector
    socp = socp.at[start:end].set(curr_socp)
    # socp = lax.dynamic_slice(
    #     soc_input, (start,), (soc_proj_sizes[i] * soc_num_proj[i],))

    # update the start point
    start = end

    new_val = socp, start
    return new_val

# val holds the vector and start point
start = 0
init_val = socp, start
val = lax.fori_loop(0, num_soc_blocks, soc_body, init_val)
socp, start = val
projection = jnp.concatenate([projection, socp])
"""
