import numpy as np
import scs
from scipy.sparse import csc_matrix


def main():
    # setup some problem
    P, A, b, c, cones_dict = robust_ls_setup(20, 20)

    scs_data = dict(P=P, A=A, b=b, c=c)

    # solve with scs
    solver = scs.SCS(scs_data,
                     cones_dict,
                     normalize=False,
                     scale=1,
                     adaptive_scale=False,
                     rho_x=1,
                     alpha=1,
                     acceleration_lookback=0,
                     eps_abs=.001,
                     eps_rel=0)
    sol = solver.solve()
    x, y, s = sol['x'], sol['y'], sol['s']
    print('x', x)
    print('y', y)
    print('s', s)

    # solve with our method


def robust_ls_setup(m_orig, n_orig):
    rho = 1

    # random A matrix
    A = (np.random.rand(m_orig, n_orig) * 2) - 1

    m_orig, n_orig = A.shape
    m, n = 2 * m_orig + n_orig + 2, n_orig + 2
    A_dense = np.zeros((m, n))
    b = np.zeros(m)

    # constraint 1
    A_dense[:n_orig, :n_orig] = -np.eye(n_orig)

    # constraint 2
    A_dense[n_orig, n_orig] = -1
    A_dense[n_orig + 1:n_orig + m_orig + 1, :n_orig] = -A

    b[n_orig:m_orig + n_orig] = np.random.normal(m_orig)  # fill in for b when theta enters --
    # here we can put anything since it will change

    # constraint 3
    A_dense[n_orig + m_orig + 1, n_orig + 1] = -1
    A_dense[n_orig + m_orig + 2:, :n_orig] = -np.eye(n_orig)

    # create sparse matrix
    A_sparse = csc_matrix(A_dense)

    # Quadratic objective
    P = np.zeros((n, n))
    P_sparse = csc_matrix(P)

    # Linear objective
    c = np.zeros(n)
    c[n_orig], c[n_orig + 1] = 1, rho

    # cones
    q_array = [m_orig + 1, n_orig + 1]
    cones_dict = dict(z=0, l=n_orig, q=q_array)
    # cones_array = jnp.array([cones["z"], cones["l"]])
    # cones_array = jnp.concatenate([cones_array, jnp.array(cones["q"])])
    return P_sparse, A_sparse, b, c, cones_dict


if __name__ == "__main__":
    main()
