import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random


def manual_vmap(predict, in_axes, out_axes, *args):
    # Preparing the arguments for vectorization
    vectorized_args = []
    for arg, axis in zip(args, in_axes):
        if axis is None:
            # If axis is None, the argument is not vectorized and is repeated for each call
            vectorized_args.append([arg] * len(args[in_axes.index(0)]))
        else:
            # If axis is 0, the argument is already a vector and is used as is
            vectorized_args.append(arg)

    # Transposing arguments to match the order needed for predict
    transposed_args = list(zip(*vectorized_args))

    # Applying predict to each set of arguments
    results = [predict(*arg_set) for arg_set in transposed_args]

    # # Handling the output axes
    # if out_axes is not None and all(isinstance(res, (list, tuple, np.ndarray)) for res in results):
    #     # If results are iterable, transpose them
    #     results = list(zip(*results))
    # # elif out_axes is not None:
    # #     # If results are not iterable (like scalars), wrap them in a list
    # #     results = [results]
    # #     # Stacking the results along the first axis if they are JAX arrays
    # elif out_axes is not None:
    #     results = jnp.stack(results, axis=0)
    if out_axes == (0):
        results = jnp.stack(results, axis=0)
    elif out_axes is not None:
        # Initialize an empty list for each output in the tuple
        stacked_results = [[] for _ in range(len(out_axes))]
        for result in results:
            for i, output in enumerate(result):
                stacked_results[i].append(output)

        # Stack each list of outputs to create JAX arrays
        for i in range(len(stacked_results)):
            stacked_results[i] = jnp.stack(stacked_results[i], axis=0)

        results = tuple(stacked_results)

    return results


def count_files_in_directory(directory):
    file_count = 0
    for _, _, files in os.walk(directory):
        file_count += len(files)
    return file_count


def setup_permutation(key_count, N_train, epochs_jit):
    permutations = []
    for i in range(epochs_jit):
        key = random.PRNGKey(key_count)
        key_count += 1
        epoch_permutation = random.permutation(key, N_train)
        permutations.append(epoch_permutation)
    stacked_permutation = jnp.stack(permutations)
    permutation = jnp.ravel(stacked_permutation)
    return permutation


def sample_plot(input, title, num_plot):
    num_plot = np.min([num_plot, 4])
    for i in range(num_plot):
        plt.plot(input[i, :])
    plt.ylabel(f"{title} values")
    plt.xlabel(f"{title} indices")
    plt.savefig(f"sample_{title}.pdf")
    plt.clf()


def vec_symm(X, triu_indices=None, factor=jnp.sqrt(2)):
    """Returns a vectorized representation of a symmetric matrix `X`.
    Vectorization (including scaling) as per SCS.
    vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)
    """

    # X = X.copy()
    X *= factor
    X = X.at[jnp.diag_indices(X.shape[0])].set(jnp.diagonal(X) / factor)
    if triu_indices is None:
        col_idx, row_idx = jnp.triu_indices(X.shape[0])
    else:
        col_idx, row_idx = triu_indices
    return X[(row_idx, col_idx)]


def unvec_symm(x, dim, triu_indices=None):
    """Returns a dim-by-dim symmetric matrix corresponding to `x`.
    `x` is a vector of length dim*(dim + 1)/2, corresponding to a symmetric
    matrix; the correspondence is as in SCS.
    X = [ X11 X12 ... X1k
              X21 X22 ... X2k
              ...
              Xk1 Xk2 ... Xkk ],
    where
    vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)
    """

    X = jnp.zeros((dim, dim))

    # triu_indices gets indices of upper triangular matrix in row-major order
    if triu_indices is None:
        col_idx, row_idx = jnp.triu_indices(dim)
    else:
        col_idx, row_idx = triu_indices
    z = jnp.zeros(x.size)

    if x.ndim > 1:
        for i in range(x.size):
            z = z.at[i].set(x[i][0, 0])
    else:
        z = x

    X = X.at[(row_idx, col_idx)].set(z)

    X = X + X.T
    X /= jnp.sqrt(2)
    X = X.at[jnp.diag_indices(dim)].set(jnp.diagonal(X) * jnp.sqrt(2) / 2)
    return X


# non jit loop
def python_fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val
