import numpy as np
import pdb
from scipy import sparse
import jax.numpy as jnp
from scipy.sparse import csc_matrix
import jax.scipy as jsp
import matplotlib.pyplot as plt
import logging
from scipy import sparse
from jax import jit, vmap

log = logging.getLogger(__name__)


def static_canon(T, nx, nu, state_box, control_box,
                 Q_val,
                 QT_val,
                 R_val,
                 Ad=None,
                 Bd=None,
                 delta_control_box=None):
    '''
    take in (nx, nu, )

    Q, R, q, QT, qT, xmin, xmax, umin, umax, T

    return (P, c, A, b) ... but b will change so not meaningful

    x0 is always the only thing that changes!
    (P, c, A) will be the same
    (b) will change in the location where x_init is!
    '''

    # keep the following data the same for all
    if isinstance(Q_val, int) or isinstance(Q_val, float):
        Q = Q_val * np.eye(nx)
    else:
        Q = np.diag(Q_val)
    if isinstance(QT_val, int) or isinstance(QT_val, float):
        QT = QT_val * np.eye(nx)
    else:
        QT = np.diag(QT_val)
    if isinstance(R_val, int) or isinstance(R_val, float):
        R = R_val * np.eye(nu)
    else:
        R = np.diag(R_val)

    q = np.zeros(nx)  # np.random.normal(size=(nx))#
    qT = np.zeros(nx)
    # Ad = np.random.normal(size=(nx, nx))
    # Bd = np.random.normal(size=(nx, nu))
    if Ad is None and Bd is None:
        Ad = .1 * np.random.normal(size=(nx, nx))
        Bd = .1 * np.random.normal(size=(nx, nu))

    '''
    umin = xmin = -1
    umax = xmax = +1
    '''

    # Quadratic objective
    P_sparse = sparse.block_diag(
        [sparse.kron(sparse.eye(T-1), Q), QT, sparse.kron(sparse.eye(T), R)],
        format="csc",
    )

    # Linear objective
    c = np.hstack([np.kron(np.ones(T-1), q), qT, np.zeros(T * nu)])

    # Linear dynamics
    Ax = sparse.kron(sparse.eye(T + 1), -sparse.eye(nx)) + sparse.kron(
        sparse.eye(T + 1, k=-1), Ad
    )
    Ax = Ax[nx:, nx:]

    Bu = sparse.kron(
        sparse.eye(T), Bd
    )
    Aeq = sparse.hstack([Ax, Bu])

    beq = np.zeros(T * nx)
    # update the first nx entries of beq to be A@x_init

    '''
    top block for (x, u) <= (xmax, umax)
    bottom block for (x, u) >= (xmin, umin)
    i.e. (-x, -u) <= (-xmin, -umin)
    '''
    # if delta_control_box is None:
    #     A_ineq = sparse.vstack(
    #         [sparse.eye(T * nx + T * nu),
    #          -sparse.eye(T * nx + T * nu)]
    #     )
    # else:
    #     A_delta_u = sparse.kron(sparse.eye(T), -sparse.eye(nu)) + sparse.kron(
    #         sparse.eye(T, k=-1), sparse.eye(nu)
    #     )
    #     A_ineq = sparse.vstack(
    #         [sparse.eye(T * nx + T * nu),
    #          -sparse.eye(T * nx + T * nu),
    #          A_delta_u,
    #          -A_delta_u]
    #     )
    if state_box == np.inf:
        zero_states = csc_matrix((T * nu, T * nx))
        block1 = sparse.hstack([zero_states, sparse.eye(T * nu)])
        block2 = sparse.hstack([zero_states, -sparse.eye(T * nu)])
        A_ineq = sparse.vstack(
            [block1,
             block2]
        )
    else:
        A_ineq = sparse.vstack(
            [sparse.eye(T * nx + T * nu),
             -sparse.eye(T * nx + T * nu)]
        )
    if delta_control_box is not None:
        A_delta_u = sparse.kron(sparse.eye(T), -sparse.eye(nu)) + sparse.kron(
            sparse.eye(T, k=-1), sparse.eye(nu)
        )
        zero_states = csc_matrix((T * nu, T * nx))
        block1 = sparse.hstack([zero_states, A_delta_u])
        block2 = sparse.hstack([zero_states, -A_delta_u])
        A_ineq = sparse.vstack([A_ineq, block1, block2])

    # stack A
    A_sparse = sparse.vstack(
        [
            Aeq,
            A_ineq
        ]
    )

    if isinstance(control_box, int) or isinstance(control_box, float):
        control_lim = control_box * np.ones(T * nu)
    else:
        control_lim = control_box

    # get b
    if state_box == np.inf:
        # b_control_box = np.kron(control_lim, np.ones(T))
        b_control_box = np.kron(np.ones(T), control_lim)
        b_upper = np.hstack(
            [b_control_box])
        b_lower = np.hstack(
            [b_control_box])
    else:
        b_upper = np.hstack(
            [state_box*np.ones(T * nx), control_lim])
        b_lower = np.hstack(
            [state_box*np.ones(T * nx), control_lim])
    if delta_control_box is None:
        b = np.hstack([beq, b_upper, b_lower])
    else:
        if isinstance(delta_control_box, int) or isinstance(delta_control_box, float):
            delta_control_box_vec = delta_control_box * np.ones(nu)
        else:
            delta_control_box_vec = delta_control_box
        b_delta_u = np.kron(delta_control_box_vec, np.ones(T))
        b = np.hstack([beq, b_upper, b_lower, b_delta_u, b_delta_u])

    # cones = dict(z=T * nx, l=2 * (T * nx + T * nu))
    num_ineq = b.size - T * nx
    cones = dict(z=T * nx, l=num_ineq)
    cones_array = jnp.array([cones['z'], cones['l']])

    # create the matrix M
    m, n = A_sparse.shape
    M = jnp.zeros((n + m, n + m))
    P = P_sparse.todense()
    A = A_sparse.todense()
    P_jax = jnp.array(P)
    A_jax = jnp.array(A)
    M = M.at[:n, :n].set(P_jax)
    M = M.at[:n, n:].set(A_jax.T)
    M = M.at[n:, :n].set(-A_jax)

    # factor for DR splitting
    algo_factor = jsp.linalg.lu_factor(M + jnp.eye(n+m))

    A_sparse = csc_matrix(A)
    P_sparse = csc_matrix(P)

    out_dict = dict(M=M,
                    algo_factor=algo_factor,
                    cones_array=cones_array,
                    A_sparse=A_sparse,
                    P_sparse=P_sparse,
                    b=b,
                    c=c,
                    A_dynamics=Ad)

    return out_dict
