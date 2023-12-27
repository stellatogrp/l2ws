import functools

import jax.numpy as jnp
import numpy as np
from jax import jit, random, vmap
from scipy.spatial import distance_matrix


def compute_weight_norm_squared(nn_params):
    weight_norms = np.zeros(len(nn_params))
    nn_weights = nn_params
    num_weights = 0
    for i, params in enumerate(nn_weights):
        weight_matrix, bias_vector = params
        weight_norms[i] = jnp.linalg.norm(weight_matrix) ** 2 + jnp.linalg.norm(bias_vector) ** 2
        num_weights += weight_matrix.size + bias_vector.size
    return weight_norms.sum(), num_weights

def compute_KL_penalty(nn_params, post_sigma, prior_sigma):
                    #    post_sigma_nn, post_sigma_beta, 
                    #    prior_sigma_nn, prior_sigma_beta):
    # num_weights = get_num_weights(nn_params)
    
    weight_norm_squared, num_weights = compute_weight_norm_squared(nn_params)
    print('weight_norm_squared', weight_norm_squared)
    kl_nn = compute_subset_KL_penalty(num_weights, weight_norm_squared, 
                                      post_sigma, prior_sigma)
    # kl_beta = compute_subset_KL_penalty(1, beta[0][0][0] ** 2, 
    #                                     post_sigma_beta, prior_sigma_beta)
    return kl_nn #+ kl_beta


def compute_subset_KL_penalty(d, weight_norm_squared, post_sigma, prior_sigma):
    return .5 * (weight_norm_squared / prior_sigma + \
                 d * (-1 + post_sigma / prior_sigma + np.log(prior_sigma / post_sigma)))


def get_perturbed_weights(key, sizes, sigma):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, sigma) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def get_nearest_neighbors(train_inputs, test_inputs, z_stars_train):
    distances = distance_matrix(np.array(test_inputs), np.array(train_inputs))
    indices = np.argmin(distances, axis=1)
    np.min(distances, axis=1)

    # print('distances', distances)
    # print('indices', indices)
    # print('best val', best_val)

    return z_stars_train[indices, :]


def random_layer_params(m, n, key, scale=1e-2):
# def random_layer_params(m, n, key, scale=1e-2):
# def random_layer_params(m, n, key, scale=1e-1):
    w_key, b_key = random.split(key)
    # fan_in, fan_out = shape[0], shape[1]
    # scale = jnp.sqrt(2.0 / (m + n))
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def init_matrix_params(t, n, key):
    X_list = []
    for i in range(t):
        U = random.normal(key + i, (n, n))
        X = U @ U.T
        norm_X = X / X.max()
        X_list.append(norm_X)
    return X_list


def relu(x):
    return jnp.maximum(0, x)


@jit
def predict_y(params, inputs):
    for W, b in params[:-1]:
        outputs = jnp.dot(W, inputs) + b
        inputs = relu(outputs)
    final_w, final_b = params[-1]
    outputs = jnp.dot(final_w, inputs) + final_b
    return outputs


batched_predict_y = vmap(predict_y, in_axes=(None, 0))


@functools.partial(jit, static_argnums=(1,))
def full_vec_2_components(input, T):
    L = input[0]
    L_vec = input[1:T]
    x = input[T:3*T]
    delta = input[3*T:3*T+2*(T-1)]
    s = input[3*T+2*(T-1):]
    return L, L_vec, x, delta, s
