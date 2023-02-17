import jax.numpy as jnp
from jax import lax, vmap, jit
from utils.generic_utils import vec_symm, unvec_symm
from functools import partial
import numpy as np
import pdb


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
        # num_soc = len(cones['q'])
        soc_total = sum(cones['q'])
        soc_cones_array = jnp.array(cones['q'])
        # soc_size = soc_cones_array[0]

        # calculate soc_proj_sizes, soc_num_proj
        # soc_proj_sizes, soc_num_proj = [], []
        # prev_size = soc_cones_array[0]
        # num_counter = 0
        # for i in range(soc_cones_array.size):
        #     curr_size = soc_cones_array[i]
        #     if curr_size == prev_size:
        #         num_counter += 1
        #     else:
        #         prev_size = curr_size
        #         soc_num_proj.append()
        m = jnp.r_[True, soc_cones_array[:-1] != soc_cones_array[1:], True]
        counts = jnp.diff(jnp.flatnonzero(m))
        unq = soc_cones_array[m[:-1]]
        out = jnp.c_[unq, counts]

        # need to convert to list so that the item is not a traced object
        soc_proj_sizes, soc_num_proj = out[:, 0].tolist(), out[:, 1].tolist()
        # pdb.set_trace()
    else:
        soc_total = 0
        soc_cones_array = jnp.array([])
    if sdp:
        num_sdp = len(cones['s'])
        sdp_total = sum(cones['s'])
        sdp_cones_array = jnp.array(cones['s'])
        sdp_size = int(sdp_cones_array[0] * (sdp_cones_array[0]+1) / 2)
        sdp_proj_single_dim = partial(sdp_proj_single, dim=sdp_cones_array[0])
        sdp_proj_single_batch = vmap(sdp_proj_single_dim, in_axes=(0), out_axes=(0))
        psd_size = sdp_size
    else:
        psd_size = 0
        sdp_total = 0
        num_sdp = 0
        sdp_size = 0
        sdp_proj_single_batch = None

    projection = partial(proj,
                         n=n,
                         zero_cone_int=int(zero_cone),
                         nonneg_cone_int=int(nonneg_cone),
                         soc_proj_sizes=soc_proj_sizes,
                         soc_num_proj=soc_num_proj,
                         soc_total=soc_total,
                         #  soc=soc,
                         #  soc_total=soc_total,
                         #  num_soc=num_soc,
                         #  soc_size=soc_size,
                         sdp=sdp,
                         sdp_total=sdp_total,
                         num_sdp=num_sdp,
                         sdp_size=sdp_size,
                         sdp_proj_single_batch=sdp_proj_single_batch
                         )
    return jit(projection), psd_size, sdp


# def proj(input, n, zero_cone_int, nonneg_cone_int, soc, soc_total, num_soc, soc_size, sdp,
#          sdp_total, num_sdp, sdp_size, sdp_proj_single_batch):
# def proj(input, n, zero_cone_int, nonneg_cone_int, soc_cones_array, soc_total, sdp,
#          sdp_total, num_sdp, sdp_size, sdp_proj_single_batch):
# partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
def proj(input, n, zero_cone_int, nonneg_cone_int, soc_proj_sizes, soc_num_proj, soc_total, sdp,
         sdp_total, num_sdp, sdp_size, sdp_proj_single_batch):
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
    # soc_size = soc_cones_array[0]
    # num_soc = soc_cones_array.size
    # soc_total = soc_cones_array.sum()
    # soc = num_soc > 0
    num_soc_blocks = len(soc_proj_sizes) #soc_proj_sizes.size
    soc = num_soc_blocks > 0

    # sdp setup

    if soc:
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
    if sdp:
        sdp = jnp.zeros(sdp_total)
        sdp_input = input[n + zero_cone_int+nonneg_cone_int+soc_total:]
        sdp_input_reshaped = jnp.reshape(sdp_input, (num_sdp, sdp_size))
        sdp_out_reshaped = sdp_proj_single_batch(sdp_input_reshaped)
        sdp = jnp.ravel(sdp_out_reshaped)
        projection = jnp.concatenate([projection, sdp])
    return projection


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
