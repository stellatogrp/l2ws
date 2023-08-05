from cProfile import label
import matplotlib.pyplot as plt
from pandas import read_csv
import sys
import jax.numpy as jnp
import pdb
import yaml
import os
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
import math
import matplotlib.colors as mcolors
from l2ws.utils.data_utils import recover_last_datetime
from examples.robust_kalman import multiple_random_robust_kalman
from examples.sparse_pca import multiple_random_sparse_pca
from l2ws.scs_problem import scs_jax
from l2ws.algo_steps import k_steps_train_scs, lin_sys_solve, create_M, create_projection_fn, get_scaled_vec_and_factor
import pdb
import jax
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    # "font.size": 24,
    "font.size": 16,
})


def main():
    contractive_plot()
    # linearly_regular_plot()
    # averaged_plot()


def contractive_plot():
    P, A, cones, q_mat, theta_mat = multiple_random_robust_kalman(
        N=5, T=50, gamma=.05, dt=.5, mu=2, rho=2, sigma=20, p=0, w_noise_var=.1, y_noise_var=.1)
    m, n = A.shape

    c, b = q_mat[0, :n], q_mat[0, n:]
    data = dict(P=P, A=A, c=c, b=b, cones=cones)

    # solve with our jax implementation
    # data = dict(P=P, A=A, c=c, b=b, cones=cones, x=x_ws, y=y_ws, s=s_ws)
    rho_x, scale, alpha, max_iters = 1, 1, 1, 500
    sol_hsde = scs_jax(data, hsde=True, iters=max_iters, jit=False,
                       rho_x=rho_x, scale=scale, alpha=alpha, plot=False)
    x_jax, y_jax, s_jax = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    fp_res_hsde = sol_hsde['fixed_point_residuals']
    # plt.plot(fp_res_hsde)
    beta = (fp_res_hsde[200] / fp_res_hsde[100]) ** (.01)

    deltas = [1,10]
    for delta in deltas:
        perturbed = fp_res_hsde + delta * beta ** np.arange(max_iters)
        # plt.plot(perturbed)
        plt.plot((perturbed - fp_res_hsde) / fp_res_hsde)
    plt.yscale('log')
    plt.title('contractive perturbation bounds')
    plt.xlabel('evaluation iterations')
    plt.savefig('perturb_plots/contractive.pdf')


def linearly_regular_plot():
    P, A, cones, q_mat, theta_mat = multiple_random_robust_kalman(
        N=5, T=50, gamma=.05, dt=.5, mu=2, rho=2, sigma=20, p=0, w_noise_var=.1, y_noise_var=.1)
    m, n = A.shape

    c, b = q_mat[0, :n], q_mat[0, n:]
    data = dict(P=P, A=A, c=c, b=b, cones=cones)

    # solve with our jax implementation
    # data = dict(P=P, A=A, c=c, b=b, cones=cones, x=x_ws, y=y_ws, s=s_ws)
    rho_x, scale, alpha, max_iters = 1, 1, 1, 210
    sol_hsde = scs_jax(data, hsde=True, iters=max_iters, jit=False,
                       rho_x=rho_x, scale=scale, alpha=alpha, plot=False)
    x_jax, y_jax, s_jax = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']
    fp_res_hsde = sol_hsde['fixed_point_residuals']
    # plt.plot(fp_res_hsde)
    beta = (fp_res_hsde[200] / fp_res_hsde[100]) ** (.01)

    deltas = [.1,1]
    for delta in deltas:
        perturbed = np.zeros(max_iters)
        gen = fp_res_hsde + delta #* beta ** np.arange(max_iters)
        perturbed[0] = gen[0]
        for i in range(1, max_iters):
            if perturbed[i - 1] * beta < gen[i]:
                print('switch', i)
            perturbed[i] = np.min([perturbed[i - 1] * beta, gen[i]])
        plt.plot((perturbed - fp_res_hsde) / fp_res_hsde)
    plt.yscale('log')
    plt.title('linearly regular perturbation bounds')
    plt.xlabel('evaluation iterations')
    plt.savefig('perturb_plots/lin_reg.pdf')


def averaged_plot():
    # sparse_pca setup
    P, A, cones, q_mat, theta_mat_jax, A_tensor = multiple_random_sparse_pca(
        n_orig=30, k=10, r=10, N=5)
    m, n = A.shape
    c, b = q_mat[0, :n], q_mat[0, n:]
    
    rho_x, scale, alpha, max_iters = 1, 1, 1, 500

    zero_cone_size = cones['z']

    M = create_M(P, A)
    hsde = True

    factor, scale_vec = get_scaled_vec_and_factor(M, rho_x, scale, m, n, zero_cone_size,
                                                       hsde=hsde)
    q_r = lin_sys_solve(factor, q_mat[0, :])
    proj = create_projection_fn(cones, n)

    # get distance to optimality and also fixed point residual
    # solve with our jax implementation
    # data = dict(P=P, A=A, c=c, b=b, cones=cones)
    supervised = False
    jit = True
    z_star = None
    z0 = jnp.ones(m + n + 1)
    q = q_mat[0, :]
    eval_out = k_steps_train_scs(2000, z0, q_r, factor, supervised, z_star, proj, jit, hsde, m, n, zero_cone_size)
    z_star, iter_losses = eval_out

    supervised = True
    z0 = jnp.ones(m + n + 1)
    eval_out_sup = k_steps_train_scs(max_iters, z0, q_r, factor, supervised, z_star, proj, jit, hsde, m, n, zero_cone_size)
    z_final, opt_losses = eval_out_sup

    supervised = False
    eval_out = k_steps_train_scs(max_iters, z0, q_r, factor, supervised, z_star, proj, jit, hsde, m, n, zero_cone_size)
    z_star, iter_losses = eval_out
    import pdb
    pdb.set_trace()

    # x_jax, y_jax, s_jax = sol_hsde['x'], sol_hsde['y'], sol_hsde['s']


def create_toy_example():
    """proximal Gradient descent for 

    min (x_1 - a)^2 + (x_2 - b)^2

    z_hist = run_prox_gd(init)
    """
    def f(x):
        return (x[0] - a) ** 2 + (x[1] - b) ** 2
    
    grad = jax.grad(f)

    # setup x_inits
    x_inits = [jnp.array([-2, -2]), jnp.array([2, 2])]
    step_size, num_steps = 0.01, 20

    x_hists = []
    for i in range(len(x_inits)):
        x_init = x_inits[i]
        x_hist = run_prox_gd(x_init, grad, step_size, num_steps)
        x_hists.append(x_hist)


    # x = np.array([gamma, 1])
    # x_hist = [x]
    # for k in range(1000):
    #     if np.linalg.norm(df(x)) < 1e-02:
    #         print("t = %.2e, converged in %d iterations" % (t, k))
    #         break
    #     dx = -df(x)
    #     #  x = line_search(x, dx)
    #     x = x + t * dx
    #     x_hist.append(x)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 9))

    # Contour
    # cs = plt.contour(X1, X2, f_vec(X1, X2), colors='k')
    #  ax.clabel(cs, fontsize=18, inline=True)

    # Gradient descent
    ax.plot(*zip(*x_hist), linestyle='--', marker='o',
            markersize=10, markerfacecolor='none', color='k')

    # Optimal solution
    # ax.scatter(*zip(x_star), marker='*', s=600, color='k')
    

    # ax.set_xlim([x_min, x_max])
    # ax.set_ylim([y_min, y_max])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    # plt.savefig("gradient_descent_%.4f.pdf" % t)
    plt.show()


def run_prox_gd(x_init, grad, step_size, num_steps):
    x_hist = []
    x = x_init
    for i in range(num_steps):
        x = jnp.clip(x - step_size * grad(x), a_min=0)
        x_hist.append(x)
    return x_hist


if __name__ == '__main__':
    main()