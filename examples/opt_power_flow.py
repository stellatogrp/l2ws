import numpy as np
import logging
import yaml
from jax import vmap
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import time
import jax.numpy as jnp
import os
import scs
import cvxpy as cp
import jax.scipy as jsp
import jax.random as jra
from l2ws.algo_steps import create_M
from scipy.sparse import csc_matrix
from examples.solve_script import setup_script
from l2ws.launcher import Workspace
from l2ws.algo_steps import get_scaled_vec_and_factor
from examples.opf_data.case14 import case14
from examples.opf_data.case30 import case30
from examples.opf_data.case57 import case57


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
    }
)
log = logging.getLogger(__name__)


def run(run_cfg):
    example = "opt_power_flow"
    data_yaml_filename = 'data_setup_copied.yaml'

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    ######################### TODO OPF MODIFY
    # set the seed
    np.random.seed(setup_cfg['seed'])
    # n_orig = setup_cfg['n_orig']
    # d_mul = setup_cfg['d_mul']

    case_num = setup_cfg['case']
    demand_vertices = setup_cfg['demand_vertices']
    d_vertices = [int(d) for d in demand_vertices]
    d_bounds = setup_cfg['demand_bounds']
    print('case number:', case_num)
    if case_num == 14:
        ppc = case14()
    elif case_num == 30:
        ppc = case30()
    elif case_num == 57:
        ppc = case57()
    else:
        print('case_num is invalid')
        exit(0)

    # k = setup_cfg['k']
    # static_dict = static_canon(n_orig, d_mul)
    # def static_canon(ppc, d_vertices, d_bounds, rho_x=1, scale=1, factor=True, seed=42):
    static_dict = static_canon(ppc, d_vertices, d_bounds)

    # non-identity DR scaling
    rho_x = run_cfg.get('rho_x', 1)
    scale = run_cfg.get('scale', 1)

    # we directly save q now
    get_q = None
    static_flag = True
    workspace = Workspace(run_cfg, static_flag, static_dict, example, get_q)

    # run the workspace
    workspace.run()


def multiple_random_opt_power_flow(ppc, d_vertices, d_bounds, N):
    out_dict = static_canon(ppc, d_vertices, d_bounds)
    # print(out_dict['prob'], out_dict['d_param'])
    # # c, b = out_dict['c'], out_dict['b']
    P_sparse, A_sparse = out_dict['P_sparse'], out_dict['A_sparse']
    cones = out_dict['cones_dict']
    prob, d_param = out_dict['prob'], out_dict['d_param']
    P, A = jnp.array(P_sparse.todense()), jnp.array(A_sparse.todense())

    # get theta_mat and d_vals together
    d_matrix = generate_theta_mat_d_vals(N, d_param, d_bounds)
    theta_mat_jax = jnp.array(d_matrix)

    # convert to q_mat
    m, n = A.shape
    q_mat = get_q_mat(d_matrix, prob, d_param, m, n)

    return P, A, cones, q_mat, theta_mat_jax # possibly return more


def generate_graph_and_prob(ppc, d_vertices, d_bounds, seed=42):
    # print('here')
    nodes = ppc['bus']
    edges = ppc['branch']
    gen = ppc['gen']
    node_index = 0
    vmin_index = 12
    vmax_index = 11
    r_index = 2
    x_index = 3
    b_index = 4
    in_index = 0
    out_index = 1
    gen_pg_index = 1
    gen_qmax_index = 3
    generators = {}
    generators_qmax = {}

    for row in gen:
        generators[int(row[node_index])] = row[gen_pg_index]
        generators_qmax[int(row[node_index])] = row[gen_qmax_index]

    # print('generators:', generators)
    # print(generators_qmax)
    vmin_arr = []
    vmax_arr = []
    G = nx.DiGraph()

    for row in nodes:
        vmin = row[vmin_index]
        vmax = row[vmax_index]
        node_p_in = cp.Variable()
        node_p_out = cp.Variable()
        node_q_in = cp.Variable()
        node_q_out = cp.Variable()
        G.add_node(int(row[node_index]), vmin=vmin, vmax=vmax, 
                   node_p_in=node_p_in, node_p_out=node_p_out, node_q_in=node_q_in, node_q_out=node_q_out)
        vmin_arr.append(vmin)
        vmax_arr.append(vmax)
    vmin_arr = np.array(vmin_arr)
    vmax_arr = np.array(vmax_arr)

    for row in edges:
        u = int(row[in_index])
        v = int(row[out_index])
        r = row[r_index]
        x = row[x_index]
        b = row[b_index]
        if np.abs(r) >= 1e-5:
            g = (1 + b * x) / r
        else:
            g = 0
        p_var = cp.Variable()
        q_var = cp.Variable()
        G.add_edge(u, v, r=r, x=x, b=b, g=g, p_var=p_var, q_var=q_var)

    n = G.number_of_nodes()
    non_generators = G.nodes() - generators
    demand_vertices = set(d_vertices)
    other_vertices = non_generators - demand_vertices

    g_dict = nx.get_edge_attributes(G, 'g')
    b_dict = nx.get_edge_attributes(G, 'b')
    edge_p_dict = nx.get_edge_attributes(G, 'p_var')
    edge_q_dict = nx.get_edge_attributes(G, 'q_var')

    node_pin_dict = nx.get_node_attributes(G, 'node_p_in')
    node_pout_dict = nx.get_node_attributes(G, 'node_p_out')
    node_qin_dict = nx.get_node_attributes(G, 'node_q_in')
    node_qout_dict = nx.get_node_attributes(G, 'node_q_out')

    X = cp.Variable((n, n), hermitian=True)
    num_generators = len(d_vertices)
    demand_param = cp.Parameter(num_generators)
    # print(demand_param, demand_param.shape)
    obj = 0
    constraints = [
        X >> 0,
        cp.diag(cp.real(X)) >= np.square(vmin_arr),
        cp.diag(cp.real(X)) <= np.square(vmax_arr),
    ]

    for edge in G.edges:
        i, j = int(edge[0]) - 1, int(edge[1]) - 1
        gij = g_dict[edge]
        bij = b_dict[edge]
        pij_var = edge_p_dict[edge]
        qij_var = edge_q_dict[edge]
        obj += gij * (X[i, i] + X[j, j] - X[i, j] - X[j, i])

        constraints += [
            pij_var + 1j * qij_var == (X[i, i] - X[i, j]) * (gij + 1j * bij),
        ]

    for node in G.nodes:
        in_edges = G.in_edges(node)
        out_edges = G.out_edges(node)
        # print('node:', node)
        # print('in/out:', in_edges, out_edges)
        p_in = node_pin_dict[node]
        p_out = node_pout_dict[node]
        q_in = node_qin_dict[node]
        q_out = node_qout_dict[node]
        # print(p_in, p_out, q_in, q_out)

        p_in_expr = 0
        p_out_expr = 0
        q_in_expr = 0
        q_out_expr = 0
        for edge in in_edges:
            # print(edge)
            p_in_expr += edge_p_dict[edge]
            q_in_expr += edge_q_dict[edge]

        for edge in out_edges:
            p_out_expr += edge_p_dict[edge]
            q_out_expr += edge_q_dict[edge]
            # print(edge_p_dict[edge])
            # print(edge)

        constraints += [
            p_in == p_in_expr,
            q_in == q_in_expr,
            p_out == p_out_expr,
            q_out == q_out_expr,
        ]
    
    for node in generators:
        g_max = generators[node]
        q_max = generators_qmax[node]
        # print(node, g_max)
        p_in = node_pin_dict[node]
        p_out = node_pout_dict[node]

        constraints += [
            p_out - p_in <= g_max,
        ]

    for node in other_vertices:
        p_in = node_pin_dict[node]
        p_out = node_pout_dict[node]
        q_in = node_qin_dict[node]
        q_out = node_qout_dict[node]

        constraints += [
            p_in - p_out == 0,
            q_in - q_out == 0,
        ]

    # this is where the parameterized demand comes in
    for i, node in enumerate(demand_vertices):
        p_in = node_pin_dict[node]
        p_out = node_pout_dict[node]

        constraints += [
            p_in - p_out == demand_param[i],
        ]

    prob = cp.Problem(cp.Minimize(cp.real(obj)), constraints)
    # demand_param.value = np.array([.5, .5])
    # prob.solve(solver=cp.SCS, verbose=True)

    return G, prob, demand_param


def cvxpy_prob(ppc, d_vertices, d_bounds, seed=42):
    G, prob, d_param = generate_graph_and_prob(ppc, d_vertices, d_bounds)
    return prob, G, d_param


def generate_theta_mat_d_vals(N, d_param, d_bounds):
    # d = n_orig * d_mul
    # b_matrix = np.zeros((N, d))
    # # n_orig_choose_2 = int(n_orig * (n_orig + 1) / 2)
    # # theta_mat = np.zeros((N, n_orig_choose_2), dtype='complex_')
    # for i in range(N):
    #     # this is where the parameterization comes in
    #     # could modify where the xi comes from
    #     negate1 = np.random.binomial(n=1, p=0.5, size=(n_orig))
    #     negate2 = np.random.binomial(n=1, p=0.5, size=(n_orig))
    #     negate1[negate1 == 0] = -1
    #     negate2[negate2 == 0] = -1

    #     xi = np.multiply(np.random.normal(size=(n_orig), loc=x_mean, scale=np.sqrt(x_var)), negate1) \
    #         + 1j * np.multiply(np.random.normal(size=(n_orig), loc=x_mean, scale=np.sqrt(x_var)), negate2)
    #     Xi = np.outer(xi, xi.conjugate())
    #     # col_idx, row_idx = np.triu_indices(n_orig)
    #     # theta_mat[i, :] = Xi[(col_idx, row_idx)]
    #     for j in range(d):
    #         # the trace will be real for hermitian matrices, but we use np.real to remove small complex floats
    #         b_matrix[i, j] = np.real(np.trace(A_tensor[j] @ Xi))
    d_dim = d_param.shape[0]
    # print(d_dim, d_param, d_bounds)
    low, high = d_bounds[0], d_bounds[1]
    d_matrix = np.random.uniform(low=low, high=high, size=(N, d_dim))
    # print(d_matrix)
    # exit(0)
    return d_matrix


def get_q_mat(d_matrix, prob, d_param, m, n):
    """
    change this so that d_matrix, d_param is passed in
        instead of A_tensor, A_param

    I think this should work now
    """
    N = d_matrix.shape[0]
    q_mat = jnp.zeros((N, m + n))
    for i in range(N):
        # set the parameter
        d_param.value = d_matrix[i, :]

        # get the problem data
        data, _, __ = prob.get_problem_data(cp.SCS)

        c, b = data['c'], data['b']
        n = c.size
        q_mat = q_mat.at[i, :n].set(c)
        q_mat = q_mat.at[i, n:].set(b)
    return q_mat


def static_canon(ppc, d_vertices, d_bounds, rho_x=1, scale=1, factor=True, seed=42):
    # create the cvxpy problem
    prob, G, d_param = cvxpy_prob(ppc, d_vertices, d_bounds, seed=42)
    # get the problem data
    data, _, __ = prob.get_problem_data(cp.SCS)

    A_sparse, c, b = data['A'], data['c'], data['b']
    m, n = A_sparse.shape
    P_sparse = csc_matrix(np.zeros((n, n)))
    cones_cp = data['dims']

    # factor for DR splitting
    m, n = A_sparse.shape
    P_jax, A_jax = jnp.array(P_sparse.todense()), jnp.array(A_sparse.todense())
    M = create_M(P_jax, A_jax)
    zero_cone_size = cones_cp.zero

    if factor:
        algo_factor, scale_vec = get_scaled_vec_and_factor(M, rho_x, scale, m, n,
                                                           zero_cone_size)
        # algo_factor = jsp.linalg.lu_factor(M + jnp.eye(n + m))
    else:
        algo_factor = None

    # set the dict
    cones = {'z': cones_cp.zero, 'l': cones_cp.nonneg, 'q': cones_cp.soc, 's': cones_cp.psd}
    out_dict = dict(
        M=M,
        algo_factor=algo_factor,
        cones_dict=cones,
        A_sparse=A_sparse,
        P_sparse=P_sparse,
        b=b,
        c=c,
        prob=prob,
        # A_param=A_param,
        G=G,
        d_param=d_param,
    )
    return out_dict


def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test

    case_num = cfg.case
    demand_vertices = cfg.demand_vertices
    d_vertices = [int(d) for d in demand_vertices]
    d_bounds = cfg.demand_bounds
    print('case number:', case_num)
    if case_num == 14:
        ppc = case14()
    elif case_num == 30:
        ppc = case30()
    elif case_num == 57:
        ppc = case57()
    else:
        print('case_num is invalid')
        exit(0)

    np.random.seed(cfg.seed)
    key = jra.PRNGKey(cfg.seed)

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    ################## TODO add extra params to generation
    P, A, cones, q_mat, theta_mat_jax = multiple_random_opt_power_flow(
        ppc, d_vertices, d_bounds, N)

    P_sparse, A_sparse = csc_matrix(P), csc_matrix(A)
    m, n = A.shape

    # create scs solver object
    #    we can cache the factorization if we do it like this
    b_np, c_np = np.array(q_mat[0, n:]), np.array(q_mat[0, :n])
    data = dict(P=P_sparse, A=A_sparse, b=b_np, c=c_np)
    tol_abs = cfg.solve_acc_abs
    tol_rel = cfg.solve_acc_rel
    solver = scs.SCS(data, cones, eps_abs=tol_abs, eps_rel=tol_rel)

    setup_script(q_mat, theta_mat_jax, solver, data, cones, output_filename, solve=cfg.solve)
