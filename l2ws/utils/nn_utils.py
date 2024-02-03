import functools

import cvxpy as cp
import jax.numpy as jnp
import numpy as np
from jax import jit, random, vmap
from scipy.spatial import distance_matrix



def invert_kl(q, c):
    """
    given scalars q and c returns
    kl^{-1}(q ||c) = sup p s.t. 0 <= p <= 1, KL(q || p) <= c
    """
    p_bernoulli = cp.Variable(2)
    q_bernoulli = np.array([q, 1-q])
    constraints = [c >= cp.sum(cp.kl_div(q_bernoulli,p_bernoulli)), 
                   0 <= p_bernoulli[0], p_bernoulli[0] <= 1, p_bernoulli[1] == 1.0 - p_bernoulli[0]]

    prob = cp.Problem(cp.Maximize(p_bernoulli[0]), constraints)

    # Solve problem
    prob.solve(verbose=False, solver=cp.SCS) # solver=cvx.ECOS
    
    kl_inv = p_bernoulli.value[0] 

    return kl_inv


def calculate_avg_posterior_var(params):
    sigma_params = params[1]
    if isinstance(sigma_params, tuple):
        flattened_params = jnp.concatenate([jnp.ravel(sigma_params[0]), jnp.ravel(sigma_params[1])])
        variances = jnp.exp(flattened_params)
    else:
        flattened_params = jnp.concatenate([jnp.ravel(weight_matrix) for weight_matrix, _ in sigma_params] + 
                                        [jnp.ravel(bias_vector) for _, bias_vector in sigma_params])
        variances = jnp.exp(flattened_params)
    avg_posterior_var = variances.mean()
    stddev_posterior_var = variances.std()
        # posterior_variances = []
        # for i, params in enumerate(sigma_params):
        #     posterior_variances
    return avg_posterior_var, stddev_posterior_var


def calculate_pinsker_penalty(N_train, params, c, b, delta, prior=0):
    penalty_loss = calculate_total_penalty(N_train, params, c, b, delta, prior=prior)
    return jnp.sqrt(penalty_loss / 2)


def calculate_total_penalty(N_train, params, c, b, delta, prior=0):
    pi_pen = jnp.log(jnp.pi ** 2 * N_train / (6 * delta))
    log_pen = 2 * jnp.log(b * jnp.log(c / jnp.exp(params[2])))
    penalty_loss = compute_all_params_KL(params[0], params[1], 
                                         params[2], prior=prior) + pi_pen + log_pen
    return penalty_loss /  N_train


def compute_weight_norm_squared(nn_params):
    if isinstance(nn_params, tuple):
        weight_norms = np.zeros(len(nn_params))
        weight_norms[0] = jnp.linalg.norm(nn_params[0]) ** 2
        weight_norms[1] = jnp.linalg.norm(nn_params[1]) ** 2
        num_weights = weight_norms[0].size + weight_norms[1].size
        return weight_norms.sum(), num_weights
    elif isinstance(nn_params, list):
        weight_norms = np.zeros(len(nn_params))
        nn_weights = nn_params
        num_weights = 0
        for i, params in enumerate(nn_weights):
            weight_matrix, bias_vector = params
            weight_norms[i] = jnp.linalg.norm(weight_matrix) ** 2 + jnp.linalg.norm(bias_vector) ** 2
            num_weights += weight_matrix.size + bias_vector.size
        return weight_norms.sum(), num_weights
    else:
        return jnp.linalg.norm(nn_params) ** 2, nn_params.size

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



def compute_all_params_KL(mean_params, sigma_params, lambd, prior=None):
    total_pen = 0

    if isinstance(mean_params, list):
        for i, params in enumerate(mean_params):
            weight_matrix, bias_vector = params
            weight_sigma, bias_sigma = sigma_params[i][0], sigma_params[i][1]
            total_pen += compute_single_param_KL(weight_matrix, 
                                                 jnp.exp(weight_sigma), jnp.exp(lambd))
            total_pen += compute_single_param_KL(bias_vector, 
                                                 jnp.exp(bias_sigma), jnp.exp(lambd))
    elif isinstance(mean_params, tuple):
        # tilista
        total_pen += compute_single_param_KL(mean_params[0], 
                                             jnp.exp(sigma_params[0]), jnp.exp(lambd))
        total_pen += compute_single_param_KL(mean_params[1], 
                                             jnp.exp(sigma_params[1]), 
                                             jnp.exp(lambd),
                                             prior=prior
                                             )
    else:
        # alista
        total_pen += compute_single_param_KL(mean_params, 
                                             jnp.exp(sigma_params), jnp.exp(lambd))
    return total_pen


def compute_single_param_KL(mean_params, sigma_params, lambd, prior=0):
    weight_norm_squared, d = jnp.linalg.norm(mean_params - prior) ** 2, mean_params.size
    pen = weight_norm_squared / lambd - d + jnp.sum(sigma_params) / lambd + \
                                             d * jnp.log(lambd) - jnp.sum(jnp.log(sigma_params))
    return .5 * pen


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


def random_variance_layer_params(m, n, key, init_val, scale=1e-2):
    w_key, b_key = random.split(key)
    return jnp.log(init_val) + scale * random.normal(w_key, (n, m)), jnp.log(init_val) + scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def init_variance_network_params(sizes, init_val, key, stddev):
    keys = random.split(key, len(sizes))
    return [random_variance_layer_params(m, n, k, init_val, scale=stddev) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


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
