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
import imageio
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    # "font.size": 24,
    "font.size": 24,
})


def main():
    # contractive_plot()
    # linearly_regular_plot()
    # averaged_plot()
    create_toy_example()


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


def create_toy_example(gif=False):
    """proximal Gradient descent for 

    min (x_1 - a)^2 + (x_2 - b)^2

    z_hist = run_prox_gd(init)
    """
    a, b = 0, 0
    coeff = 10
    def f(x):
        return coeff * (x[0] - a) ** 2 + (x[1] - b) ** 2 #+ x[1] * x[0]
    
    grad = jax.grad(f)

    # setup x_inits
    init_dist = 10
    theta1 = 105 * (np.pi/180)
    theta2 = 10 * (np.pi/180)
    x_inits = [init_dist * jnp.array([-jnp.sqrt(2) / 2, -np.sqrt(2) / 2]), 
            #    init_dist * jnp.array([1.0, 0.0]), 
            #    init_dist * jnp.array([jnp.sqrt(3) / 2, 0.5]), 
               init_dist * jnp.array([np.cos(theta1), np.sin(theta1)]),
               init_dist * jnp.array([np.cos(theta2), np.sin(theta2)])]
    m = 1
    L = 20
    num_steps = 50
    num_steps_display = 5
    step_size= 2 / (m + L) #1 / coeff, 10

    x_hists = []
    fp_res_list = []
    for i in range(len(x_inits)):
        x_init = x_inits[i]
        x_hist, fp_res = run_prox_gd(x_init, grad, step_size, num_steps)
        x_hists.append(x_hist)
        fp_res_list.append(fp_res)

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
    colors = ['b', 'r', 'g']
    for i in range(len(x_hists)):
        x_hist = x_hists[i]
        ax.plot(*zip(*x_hist[:num_steps_display]), linestyle='--', marker='o',
                markersize=10, markerfacecolor='none', color=colors[i])
    circle1 = plt.Circle((0, 0), 10, color='k', fill=False)
    ax.add_patch(circle1)
    ax.set_aspect('equal', adjustable='box')

    # turn off axes and show xstar
    # plt.scatter(0, 0)
    plt.scatter(np.zeros(1), np.zeros(1), marker='x', color='black', s=1000)
    ax.text(-1.5, -1, r'$z^\star$', fontsize=48, verticalalignment='center', horizontalalignment='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("motivating_example/paths.pdf")
    plt.clf()
    # plt.show()

    # plot the fixed-point residual
    for i in range(len(fp_res_list)):
        plt.plot(fp_res_list[i])
        print(fp_res_list[i])
    plt.yscale('log')
    plt.xlabel('evaluation iterations')
    plt.ylabel('fixed-point residual')
    plt.tight_layout()
    plt.savefig("motivating_example/fp_res.pdf")
    plt.clf()



    # plot both
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    linestyles = ['--', ':', '-.']

    # first plot the path
    # colors = ['b', 'r', 'g']
    cmap = plt.cm.Set1
    colors = [cmap.colors[3], cmap.colors[2], cmap.colors[4]]

    for i in range(len(x_hists)):
        x_hist = x_hists[i]
        # axs[0].plot(*zip(*x_hist[:num_steps_display]), linestyle='--', marker='o',
        #         markersize=10, markerfacecolor='none', color=colors[i])
        axs[0].plot(*zip(*x_hist[:num_steps_display]), linestyle=linestyles[i], marker='o',
                markersize=10, markerfacecolor='none', color=colors[i])
    circle1 = plt.Circle((0, 0), 10, color='k', fill=False)
    axs[0].add_patch(circle1)
    axs[0].set_aspect('equal', adjustable='box')

    # turn off axes and show xstar
    # plt.scatter(0, 0)
    axs[0].scatter(np.zeros(1), np.zeros(1), marker='*', color='black', s=500)
    axs[0].text(-2, -1, r'$z^\star$', fontsize=24, verticalalignment='center', horizontalalignment='center')
    axs[0].axis('off')

    # fill the non-negative orthant
    x = np.linspace(-12, 12, 400)
    y = np.linspace(-12, 12, 400)
    # indices = np.any(x <= 0, y <= 0)
    # axs[0].fill_betweenx(x, -12, 12, where=indices, color='red', alpha=0.3)
    # axs[0].fill_betweenx(x, -12, 12, where=(x <= 0), color='red', alpha=0.2, edgecolor=None)
    # axs[0].fill_betweenx(x, -12, 0, where=(x >= -.1), color='red', alpha=0.2, edgecolor=None)
    y2 = np.ones(400)
    y2[:200] = 12
    y2[200:] = 0
    axs[0].fill_betweenx(x, -12, y2, color='red', alpha=0.1, edgecolor=None)

    # contour plot
    X, Y = np.meshgrid(x, y)
    X_unraveled = np.ravel(X)
    Y_unraveled = np.ravel(Y)
    XY = np.stack([X_unraveled, Y_unraveled])
    z_vec = f(XY)
    # import pdb
    # pdb.set_trace()
    z_mat = np.reshape(z_vec, (400, 400))
    levels = np.linspace(0, 1100, 6)
    # axs[0].contour(x, y, z_mat, levels=levels, colors='gray', linestyles='dotted')
    axs[0].contour(x, y, z_mat, levels=levels, cmap='Blues', linestyles='dotted')
    
    # axs[0].tight_layout()

    # next plot the fp res
    for i in range(len(fp_res_list)):
        plt.plot(fp_res_list[i], color=colors[i], linestyle=linestyles[i])
        # print(fp_res_list[i])
    axs[1].set_yscale('log')
    axs[1].set_xlabel('evaluation iterations')
    axs[1].set_ylabel('fixed-point residual')
    # axs[1].tight_layout()
    plt.tight_layout()
    plt.savefig("motivating_example/paths_and_fp_res.pdf")
    plt.clf()


    if gif:
        # create the plots at each iteration
        # if not exis
        # os.mkdir('motivating_example')
        filenames = []
        
        for j in range(len(x_hists[0])):
            fig, ax = plt.subplots(figsize=(16, 9))
            
            for i in range(len(x_hists)):
                x_hist = x_hists[i]
                # ax.plot(*zip(*x_hist), linestyle='--', marker='o',
                #         markersize=10, markerfacecolor='none', color=colors[i])
                ax.set_xlim([-11, 11])  # Set limits for the X-axis
                ax.set_ylim([-11, 11])
                ax.scatter(x_hist[j][0], x_hist[j][1], s=100, color=colors[i])
                
                        #    linestyle='--', marker='o',
                        # markersize=10, markerfacecolor='none', 
            circle1 = plt.Circle((0, 0), 10, color='k', fill=False)
            ax.add_patch(circle1)
            ax.set_aspect('equal', adjustable='box')
            filename = f"motivating_example/curr_point_{j}.jpg"
            filenames.append(filename)
            plt.savefig(filename)
            plt.clf()

        # create the gif
        with imageio.get_writer(f"motivating_example/paths.gif", mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        # remove the files
        # delete the images - todo
        for filename in set(filenames):
            os.remove(filename)

        # Optimal solution
        # ax.scatter(*zip(x_star), marker='*', s=600, color='k')
        

        # ax.set_xlim([x_min, x_max])
        # ax.set_ylim([y_min, y_max])
        # ax.set_xticks([])
        # ax.set_yticks([])
        


def run_prox_gd(x_init, grad, step_size, num_steps):
    x_hist = [x_init]
    x = x_init
    fp_res = np.zeros(num_steps)
    for i in range(num_steps):
        x_prev = x
        x = jnp.clip(x - step_size * grad(x), a_min=0)
        x_hist.append(x)
        # if i > 0:
        
        fp_res[i] = np.linalg.norm(x - x_prev)
        print(fp_res)
    return x_hist, fp_res


if __name__ == '__main__':
    main()