import jax.numpy as jnp
import matplotlib.pyplot as plt

# from l2ws.scs_problem import scs_jax
from jaxopt import Bisection
from matplotlib import pyplot as plt

# from scipy.ndimage import gaussian_filter
# from scipy.optimize import differential_evolution
from l2ws.algo_steps import kl_inv_fn
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# import imageio
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    # "font.size": 24,
    "font.size": 24,
})


def main():
    pinsker()
    plot_3d()
    



def pinsker():
    q = 0.2
    bisec = Bisection(optimality_fun=kl_inv_fn, lower=0.0, upper=0.99999999999, 
                                      check_bracket=False,
                                      jit=True)

    # q_expit = 1 / (1 + jnp.exp(-.1 * (q - 0)))
    # q_expit = 1 / (1 + jnp.exp(-1 * (q - 0)))
    # import pdb
    # pdb.set_trace()
    num = 100
    c_vals = np.linspace(0.0001, 5, num)
    pinsker = np.zeros(num)
    klinv = np.zeros(num)

    cmap = plt.cm.Set1
    colors = cmap.colors

    for i in range(num):
        out = bisec.run(q=q, c=c_vals[i])
        pinsker[i] = q + np.sqrt(c_vals[i] / 2)
        r = out.params
        klinv[i] = q + (1 - q) * r
        print(i, klinv[i])
    
    plt.plot(c_vals, pinsker, color=colors[0])
    plt.plot(c_vals, klinv, color=colors[1])
    
    # plt.show()
    plt.tight_layout()
    plt.title(r'cross section at $q=0.2$')
    plt.xlabel(r'c')
    plt.savefig("pinsker/vary_c.pdf", bbox_inches='tight')

    # plt.xlabel('q')
    # plt.savefig('pinsker/vary_c.pdf')
    plt.clf()


    q_vals = np.linspace(0.0001, 1, num)
    pinsker = np.zeros(num)
    klinv = np.zeros(num)
    c = 0.3
    for i in range(num):
        out = bisec.run(q=q_vals[i], c=c)
        pinsker[i] = q_vals[i] + np.sqrt(c)
        r = out.params
        klinv[i] = q_vals[i] + (1 - q_vals[i]) * r
        print(i, q_vals[i], klinv[i])
    
    plt.plot(q_vals, pinsker, color=colors[0])
    plt.plot(q_vals, klinv, color=colors[1])
    
    # plt.show()
    plt.tight_layout()
    plt.title(r'cross section at $c=0.3$')
    plt.xlabel(r'$q$')
    plt.savefig('pinsker/vary_q.pdf', bbox_inches='tight')


def plot_3d():
    num = 20
    q_vals = np.linspace(0.001, 1, num)  # Replace with your range of q
    c_vals = np.linspace(0.001, 5, num)  # Replace with your range of c
    Q, C = np.meshgrid(q_vals, c_vals)
    pinsker = np.zeros((num, num))
    klinv = np.zeros((num, num))
    bisec = Bisection(optimality_fun=kl_inv_fn, lower=0.0, upper=0.99999999999, 
                                      check_bracket=False,
                                      jit=True)
    for i in range(num):
        for j in range(num):
            print(i, j)
            pinsker[i, j] = q_vals[i] + np.sqrt(c_vals[j])

            out = bisec.run(q=q_vals[i], c=c_vals[j])
            r = out.params
            klinv[i ,j] = q_vals[i] + (1 - q_vals[i]) * r
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.cm.Set1
    colors = cmap.colors
    surf = ax.plot_surface(Q, C, pinsker, color=colors[0]) #, cmap='viridis')
    surf2 = ax.plot_surface(Q, C, klinv, color=colors[1])
    ax.set_xlabel(r'$q$', labelpad=10)
    ax.set_ylabel(r'$c$', labelpad=10)
    # ax.set_title('3D visualization', pad=-100)
    # ax.set_zlabel(r'$p^\star(q,c)$', labelpad=10)
    ax.view_init(elev=10, azim=-60)
    # plt.show()
    plt.tight_layout()
    plt.savefig('pinsker/pinsker_3d.pdf')


if __name__ == '__main__':
    main()