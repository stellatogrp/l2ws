import numpy as np
from scipy import sparse
import jax.numpy as jnp
from scipy.sparse import csc_matrix
import jax.scipy as jsp
import logging
from trajax import integrators
import jax

log = logging.getLogger(__name__)


def closed_loop_rollout(qp_solver, sim_len, x_init_traj, u0, dynamics, system_constants, ref_traj_dict, budget, noise_list):
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
    qp_solver: input: A, B, x0, u0, ref_traj, budget
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
        so theta = (x0, u0, ref_traj)
    """
    T, dt = system_constants['T'], system_constants['dt']
    nx, nu = system_constants['nx'], system_constants['nu']
    x_min, x_max = system_constants['x_min'], system_constants['x_max']
    u_min, u_max = system_constants['u_min'], system_constants['u_max']
    delta_u = system_constants['delta_u']

    # noise
    if noise_list is None:
        noise_list = [jnp.zeros(nx)] * sim_len

    # first state in the trajectory is given
    x0 = x_init_traj
    # u0 = jnp.array([9.8, 0, 0, 0])
    # u0 = jnp.array([9.8, 9.8, 9.8, 9.8]) / 4

    sols = []
    state_traj_list = [x0]
    P_list, A_list, factor_list, q_list = [], [], [], []
    x0_list, u0_list, x_ref_list = [], [], []
    obstacle_num = 0
    integrator = integrators.rk4(dynamics, dt=dt)
    n = T * (nx + nu)
    m = T * (2 * nx + 2 * nu)
    # m = T * (2 * nx + 2 * nu)
    prev_sol = jnp.zeros(m + n)

    u00 = jnp.array([9.8, 0, 0, 0])

    violations = 0

    for j in range(sim_len):
        # Compute the state matrix Ad
        Ac = jax.jacobian(lambda x: dynamics(x, u0, j))(x0)

        # Compute the input matrix B
        Bc = jax.jacobian(lambda u: dynamics(x0, u, j))(u0)

        print(j)

        # solve the qp
        ref_traj = get_curr_ref_traj(ref_traj_dict, j, obstacle_num)

        print('ref_traj', ref_traj)
        x_dot = dynamics(x0, u0, j)
        sol, P, A, factor, q = qp_solver(Ac, Bc, x0, u0, x_dot, ref_traj, budget, prev_sol)
        sols.append(sol)
        P_list.append(P)
        A_list.append(A)
        factor_list.append(factor)
        q_list.append(q)
        prev_sol = sol[:m + n]
        x0_list.append(x0)
        u0_list.append(u0)
        x_ref_list.append(ref_traj)

        # implement the first control
        old_u0 = u0
        u0 = extract_first_control(sol, T, nx, nu, control_num=0)
        for i in range(T):
            print('u', i, extract_first_control(sol, T, nx, nu, control_num=i))
        

        clipped_u0 = jnp.clip(u0, a_min=u_min, a_max=u_max)
        print('u0', clipped_u0)
        if jnp.linalg.norm(clipped_u0 - u0) > 1e-2:
            violations += 1
            print('control val constraint violation', violations, jnp.linalg.norm(clipped_u0 - u0))
            
            
        
        if jnp.any(old_u0 - u0 > delta_u + 1e-2) or jnp.any(-old_u0 + u0 > delta_u + 1e-2):
            violations += 1
            print('control change constraint violation', violations, old_u0 - u0)
            # import pdb
            # pdb.set_trace()
        clipped_delta_u = jnp.clip(u0 - old_u0, a_min=-delta_u, a_max=delta_u)
        clipped_u0 = clipped_delta_u + old_u0

        state_dot = dynamics(x0, clipped_u0, j)
        print('state_dot', state_dot)

        # get the next state
        x0 = integrator(x0, clipped_u0, j) + noise_list[j]
        if jnp.any(x0 > x_max) or jnp.any(x0 < x_min):
            violations += 1
            print('box constraint violation', violations)
        print('x0', x0)
        print('expected x0', sol[:nx])
        print('expected x1', sol[nx:2*nx])
        print('expected x2', sol[2*nx:3*nx])
        print('expected x3', sol[3*nx:4*nx])
        print('expected x4', sol[4*nx:5*nx])
        print('final expected state', sol[(T-1)*nx:T*nx])
        print('final expected control', sol[(T-1)*nu + T*nx: T*nu + T*nx])

        # check if the obstacle number should be updated
        obstacle_num = update_obstacle_num(x0, ref_traj_dict, j, obstacle_num)
        print('obstacle_num', obstacle_num)

        state_traj_list.append(x0)
        if obstacle_num == -1:
            obstacle_num = 0
    rollout_results = dict(state_traj_list=state_traj_list,
                           sol_list=sols,
                           P_list=P_list,
                           A_list=A_list,
                           factor_list=factor_list,
                           clu_list=q_list,
                           x0_list=x0_list,
                           u0_list=u0_list,
                           x_ref_list=x_ref_list)
    return rollout_results


def update_obstacle_num(x, ref_traj_dict, j, obstacle_num):
    """
    checks to see if we are close enough to the current obstacle
    """
    if ref_traj_dict['case'] == 'obstacle_course':
        # check if (x - x_ref)^T Q (x - x_ref) is small -- if it is, we move onto the next obstacle
        tol = ref_traj_dict['tol']
        Q = ref_traj_dict['Q']
        curr_ref = ref_traj_dict['traj_list'][obstacle_num]
        dist = jnp.sqrt((x - curr_ref).T @ Q @ (x - curr_ref))
        print('dist', dist)
        if dist <= tol:
            # check if obstacle course finished
            if obstacle_num == len(ref_traj_dict['traj_list']) - 1:
                obstacle_num = -1
            else:
                obstacle_num += 1
    return obstacle_num


def get_curr_ref_traj(ref_traj_dict, t, obstacle_num):
    if ref_traj_dict['case'] == 'fixed_path':
        return ref_traj_dict['traj_list'][t]
    elif ref_traj_dict['case'] == 'obstacle_course':
        return ref_traj_dict['traj_list'][obstacle_num]


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


def extract_first_control(sol, T, nx, nu, control_num=0):
    return sol[T * nx + control_num * nu: T * nx + (control_num + 1) * nu]


def static_canon_mpc_osqp(x_ref, x0, Ad, Bd, cd, T, nx, nu, x_min, x_max, u_min, u_max, Q, QT, R,
                          delta_u=None, u_prev=None):
    """
    given the mpc problem
    min (x_t - x_t^{ref})^T Q_T (x_t - x_t^{ref}) + sum_{i=1}^{T-1} (x_t - x_t^{ref})^T Q (x_t - x_t^{ref}) + u_t^T R u_t
        s.t. x_{t+1} = Ad x_t + Bd u_t
             x_min <= x_t <= x_max
             u_min <= u_t <= u_max
             -delta_u <= u_{t+1} - u_t <= delta_u (if delta_u is not None)

    returns (P, A, c, l, u) in the canonical osqp form

    It is possible that (x_ref, x0, Ad, Bd) change from problem to problem

    (T, nx, nu, x_min, x_max, u_min, u_max, Q_val, QT_val, R_val) should all be the same
    """
    if np.isscalar(Q):
        Q = Q * np.eye(nx)
    else:
        Q = Q
    if np.isscalar(Q):
        QT = QT * np.eye(nx)
    else:
        QT = QT
    if np.isscalar(R):
        R = R * np.eye(nu)
    else:
        R = R

    if x_ref is None:
        x_ref = np.zeros(nx)

    # Quadratic objective
    P = sparse.block_diag(
        [sparse.kron(sparse.eye(T-1), Q), QT, sparse.kron(sparse.eye(T), R)],
        format="csc",
    )

    # Linear objective
    c = np.hstack([np.kron(np.ones(T - 1), -Q @ x_ref), -QT @ x_ref, np.zeros(T * nu)])

    # Linear dynamics
    Ax = sparse.kron(sparse.eye(T + 1), -sparse.eye(nx)) + sparse.kron(
        sparse.eye(T + 1, k=-1), Ad
    )
    Ax = Ax[nx:, nx:]
    Bu = sparse.kron(
        sparse.eye(T), Bd
    )
    Aeq = sparse.hstack([Ax, Bu])
    # import pdb
    # pdb.set_trace()

    if delta_u is None:
        A_ineq = sparse.vstack(
            [sparse.eye(T * nx + T * nu)]
        )
    else:
        A_minmax = sparse.vstack(
            [sparse.eye(T * nx + T * nu)]
        )
        # A_deltau = np.eye(T * nu)
        # np.fill_diagonal(A_deltau[1:, :], -1)
        A_deltau = sparse.eye(T * nu) - sparse.kron(
            sparse.eye(T, k=-1), sparse.eye(nu)
        )

        zeros_sparse = sparse.coo_matrix(np.zeros((T * nu, T * nx)))
        # A_deltau_sparse = sparse.coo_matrix(A_deltau)

        # import pdb
        # pdb.set_trace()

        A_deltau_full = sparse.hstack([zeros_sparse, A_deltau])
        # import pdb
        # pdb.set_trace()

        A_ineq = sparse.vstack(
            [A_minmax,
             A_deltau_full]
        )

    A = sparse.vstack(
        [
            Aeq,
            A_ineq
        ]
    )

    # get l, u
    x_max_vec = np.tile(x_max, T)
    x_min_vec = np.tile(x_min, T)
    u_max_vec = np.tile(u_max, T)
    u_min_vec = np.tile(u_min, T)

    if delta_u is None:
        b_upper = np.hstack(
            [x_max_vec, u_max_vec])
        b_lower = np.hstack(
            [x_min_vec, u_min_vec])
    else:
        delta_u_tiled = np.tile(delta_u, T - 1)

        b_upper = np.hstack(
            [x_max_vec, u_max_vec, delta_u + u_prev, delta_u_tiled])
        b_lower = np.hstack(
            [x_min_vec, u_min_vec, -delta_u + u_prev, -delta_u_tiled])
        # import pdb
        # pdb.set_trace()
        # b_upper[T * (nx + nu): T * (nx + nu) + nu] = delta_u + u_prev
        # b_lower[T * (nx + nu): T * (nx + nu) + nu] = -delta_u + u_prev
        

    # set beq
    beq = np.zeros(T * nx)

    # use the initial state x0
    beq[:nx] = -Ad @ x0

    # add in cd
    cd_tiled = np.tile(cd, T)
    beq = beq - cd_tiled

    l = np.hstack([beq, b_lower])
    u = np.hstack([beq, b_upper])

    cones = dict(z=T * nx, l=2 * (T * nx + T * nu))

    out_dict = dict(cones=cones,
                    A=jnp.array(A.todense()),
                    P=jnp.array(P.todense()),
                    l=jnp.array(l),
                    u=jnp.array(u),
                    c=jnp.array(c),
                    A_dynamics=jnp.array(Ad))
    return out_dict
