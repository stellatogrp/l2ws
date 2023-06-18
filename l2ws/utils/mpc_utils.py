import numpy as np
from scipy import sparse
import jax.numpy as jnp
from scipy.sparse import csc_matrix
import jax.scipy as jsp
import logging
from trajax import integrators
import jax

log = logging.getLogger(__name__)


# def closed_loop_rollout(qp_solver, x_init_traj, nx, nu, ode, T, x_min, x_max, u_min, u_max, traj_list, budget):
def closed_loop_rollout(qp_solver, x_init_traj, dynamics, system_constants, traj_list, budget, noise_list):
    """
    Runs a closed loop rollout for a control problem where we solve an mpc problem at each iteration
        and run the first control input
    The policy is given by the qp_solver which runs a first order method (osqp or scs) with a fixed number of steps
        which we specify as the budget

    The ode gives the dynamics of the system
        - to solve the mpc problem we linearize the dynamics from the current point
        - i.e. x_dot = f(x, u)
        - want to linearize around x0, u0: use automatic differentiation

    implements a closed loop rollout of the mpc problem
    min .5 \sum_{t=0]^{T-1} (x_t - x_t^ref)^T Q (x_t - x_t^ref) + (x_T - x_T^ref)^T Q (x_T - x_T^ref)
        s.t. x_{t+1} = f(x)

    arguments
    qp_solver: input: A, B, x0, ref_traj, budget
        output: qp solution - primal and dual solutions stacked together (i.e. the z vector that is used as the fixed point)
        important: the qp_solver must already be customized to work with the lower and upper bounds of x and u
            and the cost matrices Q, QT, and R must already be set
    system_constants: dictionary that includes (T, nx, nu, x_min, x_max, u_min, u_max, dt)
        T: mpc horizon length
        nx: numger of states
        nu: number of controls
        x_min: a vector of lower bounds on the states
        x_max: a vector of upper bounds on the states
        u_min: a vector of lower bounds on the controls
        u_max: a vector of upper bounds on the controls
        dt: the discretization
    dynamics: a jax differentiable function that describes the dynamics in the form of
        x_dot = dynamics(x, u)
    traj_list: a list of vectors of states that is the reference trajectory
        the length of traj_list is the number of simulation steps to run
    budget: the number of first order steps we allowe the qp_solver to perform
    noise_list: is a list of vectors that describes the noise for the state

    the problem is parametric around (x0, u0, ref_traj)
        u0 is needed since (A, B) are linearized around the current control
    """
    T, dt = system_constants['T'], system_constants['dt']
    nx, nu = system_constants['nx'], system_constants['nu']
    # x_min, x_max = system_constants['x_min'], system_constants['x_max']
    # u_min, u_max = system_constants['u_min'], system_constants['u_max']
    sim_len = len(traj_list)

    # first state in the trajectory is given
    x0 = x_init_traj
    u0 = jnp.zeros(nu)

    sols = []
    for j in range(sim_len):
        # Compute the state matrix A
        A = jax.jacobian(lambda x: dynamics(x, u0))(x0)

        # Compute the input matrix B
        B = jax.jacobian(lambda u: dynamics(x0, u))(u0)

		# solve the qp
        sol = qp_solver(A, B, x0, traj_list[j], budget)
        sols.append(sol)
        
        # implement the first control
        u0 = extract_first_control(sol, T, nx, nu)
    
        # get the next state
        x0 = integrators.euler(dynamics, dt) + noise_list[j]
    return sols


def simulate_fwd_l2ws(sim_len, l2ws_model, k, noise_vec_list, q_init, x_init, A, Ad, Bd, T, nx, nu, prev_sol=False):
    """
    does the forward simulation

    returns
    """
    m, n = A.shape
    # get the first test_input and q_mat_test
    input = x_init
    q_mat = q_init

    opt_sols = []
    state_traj = [x_init]

    opt_sol = np.zeros(n + 2 * m)
    

    for i in range(sim_len):
        # evaluate
        if prev_sol:
            # get the shifted previous solution
            prev_z_shift = shifted_sol(opt_sol[:m + n], T, nx, nu, m, n)
            final_eval_out = l2ws_model.evaluate(
                k, prev_z_shift[None, :], q_mat[None, :], z_stars=None, fixed_ws=True, tag='test')
            # z_star = final_eval_out[1][2][0, -1, :]
        else:
            final_eval_out = l2ws_model.evaluate(
                k, input[None, :], q_mat[None, :], z_stars=None, fixed_ws=False, tag='test')
        print('loss', k, prev_sol, final_eval_out[1][0])

        # get the first control input
        # final_eval_out[1][2] will have shape (1, k, n + 2 * m)
        opt_sol = final_eval_out[1][2][0, -1, :]

        u0 = opt_sol[T * nx: T * nx + nu]

        # input the first control to get the next state and perturb it
        x_init = Ad @ x_init + Bd @ u0 + noise_vec_list[i]

        # set test_input and q_mat_test
        input = x_init
        c, l, u = q_mat[:n], q_mat[n:n + m], q_mat[n + m:]
        Ad_x_init = Ad @ x_init
        l = l.at[:nx].set(-Ad_x_init)
        u = u.at[:nx].set(-Ad_x_init)
        q_mat = q_mat.at[n:n + m].set(l)
        q_mat = q_mat.at[n + m:].set(u)

        # append to the optimal solutions
        opt_sols.append(opt_sol)

        # append to the state trajectory
        state_traj.append(x_init)

    return opt_sols, state_traj


def extract_first_control(sol, T, nx, nu):
    return sol[T * nx: T * nx + nu]


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

    q = np.zeros(nx)
    qT = np.zeros(nx)
    if Ad is None and Bd is None:
        Ad = .1 * np.random.normal(size=(nx, nx))
        Bd = .1 * np.random.normal(size=(nx, nu))

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

    '''
    top block for (x, u) <= (xmax, umax)
    bottom block for (x, u) >= (xmin, umin)
    i.e. (-x, -u) <= (-xmin, -umin)
    '''
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
                    cones_dict=cones,
                    A_sparse=A_sparse,
                    P_sparse=P_sparse,
                    b=b,
                    c=c,
                    A_dynamics=Ad)

    return out_dict
