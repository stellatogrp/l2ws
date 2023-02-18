import jax.numpy as jnp
from jax import lax, vmap, jit
from utils.generic_utils import vec_symm, unvec_symm
from functools import partial
import numpy as np
import pdb
import jax.scipy as jsp


def create_projection_fn(cones, n):
    """
    cones is a dict with keys
    z: zero cone
    l: non-negative cone
    q: second-order cone
    s: positive semidefinite cone
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
        sdp_proj_sizes, sdp_num_proj = count_num_repeated_elements(sdp_cones_array)
        # num_sdp = len(cones['s'])
        # sdp_total = sum(cones['s'])
        # sdp_cones_array = jnp.array(cones['s'])
        # sdp_size = int(sdp_cones_array[0] * (sdp_cones_array[0]+1) / 2)
        # sdp_proj_single_dim = partial(sdp_proj_single, dim=sdp_cones_array[0])
        # sdp_proj_single_batch = vmap(sdp_proj_single_dim, in_axes=(0), out_axes=(0))
        # psd_size = sdp_size
        # psd_size = sdp_proj_sizes[0]
        sdp_size = int(sdp_cones_array[0] * (sdp_cones_array[0]+1) / 2)
    else:
        psd_size = 0
        # sdp_total = 0
        # num_sdp = 0
        # sdp_size = 0
        # sdp_proj_single_batch = None
        sdp_proj_sizes, sdp_num_proj = [], []
    pdb.set_trace()
    projection = partial(proj,
                         n=n,
                         zero_cone_int=int(zero_cone),
                         nonneg_cone_int=int(nonneg_cone),
                         soc_proj_sizes=soc_proj_sizes,
                         soc_num_proj=soc_num_proj,
                         sdp_proj_sizes=sdp_proj_sizes,
                         sdp_num_proj=sdp_num_proj,
                        #  sdp=sdp,
                        #  sdp_total=sdp_total,
                        #  num_sdp=num_sdp,
                        #  sdp_size=sdp_size,
                        #  sdp_proj_single_batch=sdp_proj_single_batch
                         )
    return jit(projection), psd_size


# def lin_sys_solve(factor, rhs, static_flag):
#     if static_flag:
#         return jsp.linalg.lu_solve(factor, rhs)
#     else:
#         return factor @ rhs

def lin_sys_solve(factor, rhs):
    return jsp.linalg.lu_solve(factor, rhs)


def proj(input, n, zero_cone_int, nonneg_cone_int, soc_proj_sizes, soc_num_proj, sdp_proj_sizes,
         sdp_num_proj):
    """
    soc_proj_sizes: list of the sizes of the socp projections needed
    soc_num_proj: list of the number of socp projections needed for each size

    e.g. 50 socp projections of size 3 and 1 projection of size 100 would be
    soc_proj_sizes = [3, 100]
    soc_num_proj = [50, 1]
    """
    nonneg = jnp.clip(input[n + zero_cone_int: n + zero_cone_int + nonneg_cone_int], a_min=0)
    projection = jnp.concatenate([input[:n + zero_cone_int], nonneg])

    # soc setup
    num_soc_blocks = len(soc_proj_sizes)

    # avoiding doing inner product using jax so that we can jit
    soc_total = sum(i[0] * i[1] for i in zip(soc_proj_sizes, soc_num_proj))
    soc_bool = num_soc_blocks > 0

    # sdp setup
    num_sdp_blocks = len(sdp_proj_sizes)
    sdp_total = sum(i[0] * i[1] for i in zip(sdp_proj_sizes, sdp_num_proj))
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
            # curr_soc_input = soc_input[start:end]
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
            end = start + sdp_proj_sizes[i] * sdp_num_proj[i]

            # extract the right sdp_input
            curr_sdp_input = lax.dynamic_slice(
                sdp_input, (start,), (sdp_proj_sizes[i] * sdp_num_proj[i],))

            # reshape so that we vmap all of the sdp projections of the same size together
            curr_sdp_input_reshaped = jnp.reshape(
                curr_sdp_input, (sdp_num_proj[i], sdp_proj_sizes[i]))
            curr_sdp_out_reshaped = sdp_proj_batch(curr_sdp_input_reshaped, sdp_proj_sizes[i])
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
            # return (0.0*jnp.zeros(2), 0.0)
            return (0.0*jnp.zeros(y.size), 0.0)
        return lax.cond(s >= 0, case2a, case2b, y, s)
    return lax.cond(y_norm >= jnp.abs(s), case1_soc_proj, case2_soc_proj, y, s)


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
