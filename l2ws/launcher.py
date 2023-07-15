from jax.config import config
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from l2ws.l2ws_model import L2WSmodel
from l2ws.ista_model import ISTAmodel
from l2ws.osqp_model import OSQPmodel
from l2ws.scs_model import SCSmodel
from l2ws.eg_model import EGmodel
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax
import hydra
import time
from scipy.spatial import distance_matrix
from l2ws.algo_steps import create_projection_fn, get_psd_sizes, vec_symm, form_osqp_matrix, unvec_symm
from l2ws.utils.generic_utils import sample_plot, setup_permutation, count_files_in_directory
import scs
from scipy.sparse import csc_matrix
from functools import partial
from scipy.sparse import csc_matrix, save_npz, load_npz
from l2ws.utils.mpc_utils import closed_loop_rollout
from jax import vmap
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    "font.size": 16,
})
config.update("jax_enable_x64", True)


class Workspace:
    def __init__(self, algo, cfg, static_flag, static_dict, example,
                 traj_length=None,
                 custom_visualize_fn=None,
                 shifted_sol_fn=None,
                 closed_loop_rollout_dict=None):
        '''
        cfg is the run_cfg from hydra
        static_flag is True if the matrices P and A don't change from problem to problem
        static_dict holds the data that doesn't change from problem to problem
        example is the string (e.g. 'robust_kalman')
        '''
        self.algo = algo
        self.static_flag = static_flag
        self.example = example
        self.eval_unrolls = cfg.eval_unrolls
        self.eval_every_x_epochs = cfg.eval_every_x_epochs
        self.save_every_x_epochs = cfg.save_every_x_epochs
        self.num_samples = cfg.num_samples
        self.eval_batch_size = cfg.get('eval_batch_size', self.num_samples)
        self.pretrain_cfg = cfg.pretrain
        self.key_count = 0
        self.save_weights_flag = cfg.get('save_weights_flag', False)
        self.load_weights_datetime = cfg.get('load_weights_datetime', None)
        self.shifted_sol_fn = shifted_sol_fn
        self.plot_iterates = cfg.plot_iterates
        self.normalize_inputs = cfg.get('normalize_inputs', True)
        self.epochs_jit = cfg.epochs_jit
        self.accs = cfg.get('accuracies')

        # custom visualization
        self.init_custom_visualization(cfg, custom_visualize_fn)
        self.vis_num = 20

        # from the run cfg retrieve the following via the data cfg
        N_train, N_test = cfg.N_train, cfg.N_test
        N = N_train + N_test

        # solve C
        self.solve_c_num = cfg.get('solve_c_num', 0)
        self.rel_tols = cfg.get('rel_tols', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        self.abs_tols = cfg.get('abs_tols', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        if self.solve_c_num == 'all':
            self.solve_c_num = N_test

        # for control problems only
        self.closed_loop_rollout_dict = closed_loop_rollout_dict
        self.traj_length = traj_length
        if traj_length is not None:
            self.prev_sol_eval = True
        else:
            self.prev_sol_eval = False

        self.train_unrolls = cfg.train_unrolls

        # load the data from problem to problem
        jnp_load_obj = self.load_setup_data(example, cfg.data.datetime, N_train, N)
        thetas = jnp.array(jnp_load_obj['thetas'])
        self.thetas_train = thetas[:N_train, :]
        self.thetas_test = thetas[N_train:N, :]

        train_inputs, test_inputs = self.normalize_inputs_fn(thetas, N_train, N_test)
        self.train_inputs, self.test_inputs = train_inputs, test_inputs
        self.skip_startup = cfg.get('skip_startup', False)
        self.setup_opt_sols(algo, jnp_load_obj, N_train, N)

        # everything below is specific to the algo
        if algo == 'ista':
            # get b_mat
            self.q_mat_train = thetas[:N_train, :]
            self.q_mat_test = thetas[N_train:N, :]

            self.create_ista_model(cfg, static_dict)
        elif algo == 'osqp':
            self.create_osqp_model(cfg, static_dict)
        elif algo == 'scs':
            self.create_scs_model(cfg, static_dict)
        elif algo == 'extragradient':
            self.create_extragradient_model(cfg, static_dict)

    def create_ista_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        A, lambd = static_dict['A'], static_dict['lambd']
        ista_step = static_dict['ista_step']

        input_dict = dict(algorithm='ista',
                          supervised=cfg.supervised,
                          train_unrolls=self.train_unrolls,
                          jit=True,
                          train_inputs=self.train_inputs,
                          test_inputs=self.test_inputs,
                          b_mat_train=self.q_mat_train,
                          b_mat_test=self.q_mat_test,
                          lambd=lambd,
                          ista_step=ista_step,
                          A=A,
                          nn_cfg=cfg.nn_cfg,
                          z_stars_train=self.z_stars_train,
                          z_stars_test=self.z_stars_test,
                          )
        self.l2ws_model = ISTAmodel(input_dict)

    def create_extragradient_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        # A, lambd = static_dict['A'], static_dict['lambd']
        eg_step = static_dict['eg_step']
        f = static_dict['f']
        
        proj_X, proj_Y = static_dict['proj_X'], static_dict['proj_Y']
        m, n = static_dict['m'], static_dict['n']

        input_dict = dict(algorithm='extragradient',
                          f=f,
                          proj_X=proj_X,
                          proj_Y=proj_Y,
                          m=m,
                          n=n,
                          supervised=cfg.supervised,
                          train_unrolls=self.train_unrolls,
                          jit=True,
                          train_inputs=self.train_inputs,
                          test_inputs=self.test_inputs,
                          q_mat_train=self.q_mat_train,
                          q_mat_test=self.q_mat_test,
                          eg_step=eg_step,
                          nn_cfg=cfg.nn_cfg,
                          z_stars_train=self.z_stars_train,
                          z_stars_test=self.z_stars_test,
                          )
        self.l2ws_model = EGmodel(input_dict)

    def create_osqp_model(self, cfg, static_dict):
        if self.static_flag:
            factor = static_dict['factor']
            A = static_dict['A']
            P = static_dict['P']
            m, n = A.shape
            self.m, self.n = m, n
            rho = static_dict['rho']
            input_dict = dict(factor_static_bool=True,
                              supervised=cfg.supervised,
                              rho=rho,
                              q_mat_train=self.q_mat_train,
                              q_mat_test=self.q_mat_test,
                              A=A,
                              P=P,
                              m=m,
                              n=n,
                              factor=factor,
                              train_inputs=self.train_inputs,
                              test_inputs=self.test_inputs,
                              train_unrolls=self.train_unrolls,
                              eval_unrolls=self.eval_unrolls,
                              nn_cfg=cfg.nn_cfg,
                              z_stars_train=self.z_stars_train,
                              z_stars_test=self.z_stars_test,
                              jit=True)
        else:
            self.m, self.n = static_dict['m'], static_dict['n']
            m, n = self.m, self.n
            print('m, n', m, n)
            rho_vec = jnp.ones(m)
            l0 = self.q_mat_train[0, n: n + m]
            u0 = self.q_mat_train[0, n + m: n + 2 * m]
            print('l0', l0)
            print('u0', u0)
            rho_vec = rho_vec.at[l0 == u0].set(1000)

            t0 = time.time()

            # form matrices (N, m + n, m + n) to be factored
            nc2 = int(n * (n + 1) / 2)
            q_mat = jnp.vstack([self.q_mat_train, self.q_mat_test])
            N_train, N_test = self.q_mat_train.shape[0], self.q_mat_test[0]
            N = q_mat.shape[0]
            unvec_symm_batch = vmap(unvec_symm, in_axes=(0, None), out_axes=(0))
            P_tensor = unvec_symm_batch(q_mat[:, 2 * m + n: 2 * m + n + nc2], n)
            A_tensor = jnp.reshape(q_mat[:, 2 * m + n + nc2:], (N, m, n))
            sigma = 1
            batch_form_osqp_matrix = vmap(form_osqp_matrix, in_axes=(0, 0, None, None), out_axes=(0))

            # try batching
            cutoff = 4000
            matrices1 = batch_form_osqp_matrix(P_tensor[:cutoff, :, :], A_tensor[:cutoff, :, :], rho_vec, sigma)
            matrices2 = batch_form_osqp_matrix(P_tensor[cutoff:, :, :], A_tensor[cutoff:, :, :], rho_vec, sigma)
            # matrices = 


            # do factors
            # factors0, factors1 = self.batch_factors(self.q_mat_train)
            batch_lu_factor = vmap(jsp.linalg.lu_factor, in_axes=(0,), out_axes=(0, 0))
            factors10, factors11 = batch_lu_factor(matrices1)
            factors20, factors21 = batch_lu_factor(matrices2)
            factors0 = jnp.vstack([factors10, factors20])
            factors1 = jnp.vstack([factors11, factors21])

            t1 = time.time()
            print('batch factor time', t1 - t0)

            self.factors_train = (factors0[:N_train, :, :], factors1[:N_train, :])
            self.factors_test = (factors0[N_train:N, :, :], factors1[N_train:N, :])


            input_dict = dict(factor_static_bool=False,
                              supervised=cfg.supervised,
                              rho=rho_vec,
                              q_mat_train=self.q_mat_train,
                              q_mat_test=self.q_mat_test,
                              m=self.m,
                              n=self.n,
                              train_inputs=self.train_inputs,
                              test_inputs=self.test_inputs,
                              factors_train=self.factors_train,
                              factors_test=self.factors_test,
                              train_unrolls=self.train_unrolls,
                              eval_unrolls=self.eval_unrolls,
                              nn_cfg=cfg.nn_cfg,
                              z_stars_train=self.z_stars_train,
                              z_stars_test=self.z_stars_test,
                              jit=True)
        self.x_stars_train = self.z_stars_train[:, :self.n]
        self.x_stars_test = self.z_stars_test[:, :self.n]
        self.l2ws_model = OSQPmodel(input_dict)

    def create_scs_model(self, cfg, static_dict):
        # get_M_q = None
        # if get_M_q is None:
        #     q_mat = jnp_load_obj['q_mat']
        if self.static_flag:
            static_M = static_dict['M']

            static_algo_factor = static_dict['algo_factor']
            cones = static_dict['cones_dict']

            # call get_q_mat
            # if get_M_q is not None:
            #     q_mat = get_M_q(thetas)
            M_tensor_train, M_tensor_test = None, None
            matrix_invs_train, matrix_invs_test = None, None

            # M_plus_I = static_M + jnp.eye(n + m)
            # static_algo_factor = jsp.linalg.lu_factor(M_plus_I)
        else:
            pass
            # load the algo_factors -- check if factor or inverse
            # M_tensor, q_mat = get_M_q(thetas)

            # load the matrix invs
            # matrix_invs = jnp_load_obj['matrix_invs']

            # static_M, static_algo_factor = None, None

            # cones = static_dict['cones_dict']
            # M_tensor_train = M_tensor[:N_train, :, :]
            # M_tensor_test = M_tensor[N_train:N, :, :]
            # matrix_invs_train = matrix_invs[:N_train, :, :]
            # matrix_invs_test = matrix_invs[N_train:N, :, :]
        rho_x = cfg.get('rho_x', 1)
        scale = cfg.get('scale', 1)
        alpha_relax = cfg.get('alpha_relax', 1)

        # save cones
        self.cones = static_dict['cones_dict']

        # alternate -- load it if available (but this is memory-intensive)
        # N_train = self.train_inputs.shape[0]
        # N_test = self.test_inputs.shape[0]
        # N = N_train + N_test
        # q_mat_train = jnp.array(q_mat[:N_train, :])
        # q_mat_test = jnp.array(q_mat[N_train:N, :])

        self.M = static_M
        proj = create_projection_fn(cones, self.n)
        psd_sizes = get_psd_sizes(cones)

        self.psd_size = psd_sizes[0]

        input_dict = {'nn_cfg': cfg.nn_cfg,
                      'proj': proj,
                      'train_inputs': self.train_inputs,
                      'test_inputs': self.test_inputs,
                      'train_unrolls': self.train_unrolls,
                      'eval_unrolls': self.eval_unrolls,
                      'z_stars_train': self.z_stars_train,
                      'z_stars_test': self.z_stars_test,
                      'q_mat_train': self.q_mat_train,
                      'q_mat_test': self.q_mat_test,
                      'M_tensor_train': M_tensor_train,
                      'M_tensor_test': M_tensor_test,
                      'm': self.m,
                      'n': self.n,
                      'static_M': static_M,
                      'y_stars_train': self.y_stars_train,
                      'y_stars_test': self.y_stars_test,
                      'x_stars_train': self.x_stars_train,
                      'x_stars_test': self.x_stars_test,
                      'static_flag': self.static_flag,
                      'static_algo_factor': static_algo_factor,
                      'matrix_invs_train': matrix_invs_train,
                      'matrix_invs_test': matrix_invs_test,
                      'supervised': cfg.get('supervised', False),
                      'loss_method': cfg.get('loss_method', 'fixed_k'),
                      'pretrain_alpha': cfg.get('pretrain_alpha'),
                      'normalize_alpha': cfg.get('normalize_alpha'),
                      'plateau_decay': cfg.plateau_decay,
                      'rho_x': rho_x,
                      'scale': scale,
                      'alpha_relax': alpha_relax,
                      'zero_cone_size': cones['z'],
                      'cones': cones
                      }
        self.l2ws_model = SCSmodel(input_dict)

    def setup_opt_sols(self, algo, jnp_load_obj, N_train, N, num_plot=5):
        if algo != 'scs':
            z_stars = jnp_load_obj['z_stars']
            z_stars_train = z_stars[:N_train, :]
            z_stars_test = z_stars[N_train:N, :]
            self.plot_samples(num_plot, self.thetas_train, self.train_inputs, z_stars_train)
            self.z_stars_test = z_stars_test
            self.z_stars_train = z_stars_train

            # if algo == 'osqp':
            #     self.x_stars_train = z_stars_train[:, :self.n]
            #     self.x_stars_test = z_stars_test[:, :self.n]
        else:
            if 'x_stars' in jnp_load_obj.keys():
                x_stars = jnp_load_obj['x_stars']
                y_stars = jnp_load_obj['y_stars']
                s_stars = jnp_load_obj['s_stars']
                z_stars = jnp.hstack([x_stars, y_stars + s_stars])
                x_stars_train = x_stars[:N_train, :]
                y_stars_train = y_stars[:N_train, :]

                self.x_stars_train = x_stars[:N_train, :]
                self.y_stars_train = y_stars[:N_train, :]

                z_stars_train = z_stars[:N_train, :]
                self.x_stars_test = x_stars[N_train:N, :]
                self.y_stars_test = y_stars[N_train:N, :]
                z_stars_test = z_stars[N_train:N, :]
                self.m, self.n = y_stars_train.shape[1], x_stars_train[0, :].size
            else:
                x_stars_train, self.x_stars_test = None, None
                y_stars_train, self.y_stars_test = None, None
                z_stars_train, z_stars_test = None, None
                self.m, self.n = int(jnp_load_obj['m']), int(jnp_load_obj['n'])
            self.plot_samples_scs(num_plot, self.thetas_train, self.train_inputs,
                                  x_stars_train, y_stars_train, z_stars_train)
            self.z_stars_train = z_stars_train
            self.z_stars_test = z_stars_test

    def save_weights(self):
        nn_weights = self.l2ws_model.params

        # create directory
        if not os.path.exists('nn_weights'):
            os.mkdir('nn_weights')

        # Save each weight matrix and bias vector separately using jnp.savez
        for i, params in enumerate(nn_weights):
            weight_matrix, bias_vector = params
            jnp.savez(f"nn_weights/layer_{i}_params.npz", weight=weight_matrix, bias=bias_vector)

    def load_weights(self, example, datetime):
        # get the appropriate folder
        orig_cwd = hydra.utils.get_original_cwd()
        folder = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/nn_weights"

        # find the number of layers based on the number of files
        num_layers = count_files_in_directory(folder)

        # iterate over the files/layers
        params = []
        for i in range(num_layers):
            layer_file = f"{folder}/layer_{i}_params.npz"
            loaded_layer = jnp.load(layer_file)
            weight_matrix, bias_vector = loaded_layer['weight'], loaded_layer['bias']
            weight_bias_tuple = (weight_matrix, bias_vector)
            params.append(weight_bias_tuple)

        # store the weights as the l2ws_model params
        self.l2ws_model.params = params

    def normalize_inputs_fn(self, thetas, N_train, N_test):
        # normalize the inputs if the option is on
        N = N_train + N_test
        if self.normalize_inputs:
            col_sums = thetas.mean(axis=0)
            std_devs = thetas.std(axis=0)
            inputs_normalized = (thetas - col_sums) / std_devs #thetas.std(axis=0)
            inputs = jnp.array(inputs_normalized)
            
            # save the col_sums and std deviations
            self.normalize_col_sums = col_sums
            self.normalize_std_dev = std_devs
        else:
            inputs = jnp.array(thetas)
        train_inputs = inputs[:N_train, :]
        test_inputs = inputs[N_train:N, :]

        
        return train_inputs, test_inputs
    
    def normalize_theta(self, theta):
        normalized_input = (theta - self.normalize_col_sums) / self.normalize_std_dev
        return normalized_input

    def load_setup_data(self, example, datetime, N_train, N):
        orig_cwd = hydra.utils.get_original_cwd()
        folder = f"{orig_cwd}/outputs/{example}/data_setup_outputs/{datetime}"
        filename = f"{folder}/data_setup.npz"

        if self.static_flag:
            jnp_load_obj = jnp.load(filename)
        else:
            jnp_load_obj = jnp.load(filename)
            q_mat = jnp.array(load_npz(f"{filename[:-4]}_q.npz").todense())
            self.q_mat_train = q_mat[:N_train, :]
            self.q_mat_test = q_mat[N_train:N, :]

            # load factors
            # factors0, factors1 = jnp_load_obj['factors0'], jnp_load_obj['factors1']
            # factors = (factors0, factors1)
            # jnp_load_obj['factors'] = factors
            

            # compute factors
            # all_factors_train is a tuple with shapes ((N, n + m, n + m), (N, n + m))
            # factors0, factors1 = self.batch_factors(q_mat)


            # if we are in the dynamic case, then we need to get q from the sparse format
            # jnp_load_obj['q_mat'] = jnp.array(q_mat_sparse)

            # self.factors_train = (jnp.array(factors0[:N_train, :, :]), jnp.array(factors1[:N_train, :]))
            # self.factors_test = (jnp.array(factors0[N_train:N, :, :]), jnp.array(factors1[N_train:N, :]))

        if 'q_mat' in jnp_load_obj.keys():
            q_mat = jnp.array(jnp_load_obj['q_mat'])
            q_mat_train = q_mat[:N_train, :]
            q_mat_test = q_mat[N_train:N, :]
            self.q_mat_train, self.q_mat_test = q_mat_train, q_mat_test
        elif self.algo == 'extragradient':
            q_mat = jnp.array(jnp_load_obj['thetas'])
            q_mat_train = q_mat[:N_train, :]
            q_mat_test = q_mat[N_train:N, :]
            self.q_mat_train, self.q_mat_test = q_mat_train, q_mat_test

        # load the closed_loop_rollout trajectories
        if 'ref_traj_tensor' in jnp_load_obj.keys():
            # load all of the goals
            self.closed_loop_rollout_dict['ref_traj_tensor'] = jnp_load_obj['ref_traj_tensor']
        
        return jnp_load_obj

    def plot_samples(self, num_plot, thetas, train_inputs, z_stars):
        sample_plot(thetas, 'theta', num_plot)
        sample_plot(train_inputs, 'input', num_plot)
        if z_stars is not None:
            sample_plot(z_stars, 'z_stars', num_plot)

    def plot_samples_scs(self, num_plot, thetas, train_inputs, x_stars, y_stars, z_stars):
        sample_plot(thetas, 'theta', num_plot)
        sample_plot(train_inputs, 'input', num_plot)
        if x_stars is not None:
            sample_plot(x_stars, 'x_stars', num_plot)
            sample_plot(y_stars, 'y_stars', num_plot)
            sample_plot(z_stars, 'z_stars', num_plot)

    def init_custom_visualization(self, cfg, custom_visualize_fn):
        iterates_visualize = cfg.get('iterates_visualize', 0)
        if custom_visualize_fn is None or iterates_visualize == 0:
            self.has_custom_visualization = False
        else:
            self.has_custom_visualization = True
            self.custom_visualize_fn = custom_visualize_fn
            self.iterates_visualize = iterates_visualize

    def _init_logging(self):
        self.logf = open('log.csv', 'a')
        fieldnames = ['iter', 'train_loss', 'val_loss', 'test_loss', 'time_train_per_epoch']
        self.writer = csv.DictWriter(self.logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            self.writer.writeheader()

        self.test_logf = open('log_test.csv', 'a')
        fieldnames = ['iter', 'train_loss', 'test_loss', 'time_per_iter']
        self.test_writer = csv.DictWriter(self.test_logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            self.test_writer.writeheader()

        self.logf = open('train_results.csv', 'a')

        fieldnames = ['train_loss', 'moving_avg_train', 'time_train_per_epoch']
        self.writer = csv.DictWriter(self.logf, fieldnames=fieldnames)
        if os.stat('train_results.csv').st_size == 0:
            self.writer.writeheader()

        self.test_logf = open('train_test_results.csv', 'a')
        fieldnames = ['iter', 'train_loss', 'test_loss', 'time_per_iter']
        self.test_writer = csv.DictWriter(self.test_logf, fieldnames=fieldnames)
        if os.stat('train_results.csv').st_size == 0:
            self.test_writer.writeheader()

    def evaluate_iters(self, num, col, train=False, plot=True, plot_pretrain=False):
        if train and col == 'prev_sol':
            return
        fixed_ws = col == 'nearest_neighbor' or col == 'prev_sol'

        # do the actual evaluation (most important step in thie method)
        eval_out = self.evaluate_only(fixed_ws, num, train, col, self.eval_batch_size)

        # extract information from the evaluation
        loss_train, out_train, train_time = eval_out
        iter_losses_mean = out_train[1].mean(axis=0)
        angles = out_train[3]
        # iter_losses_mean = out_train[2].mean(axis=0)
        # angles = out_train[3]
        # primal_residuals = out_train[4].mean(axis=0)
        # dual_residuals = out_train[5].mean(axis=0)

        # plot losses over examples
        # losses_over_examples = out_train[2].T
        losses_over_examples = out_train[1].T
        self.plot_losses_over_examples(losses_over_examples, train, col)

        # update the eval csv files
        # df_out = self.update_eval_csv(
        #     iter_losses_mean, primal_residuals, dual_residuals, train, col)
        # df_out = self.update_eval_csv(
        #     iter_losses_mean, train, col)
        primal_residuals, dual_residuals, obj_vals_diff = None, None, None
        if len(out_train) == 6 or len(out_train) == 8:
            primal_residuals = out_train[4].mean(axis=0)
            dual_residuals = out_train[5].mean(axis=0)
        elif len(out_train) == 5:
            obj_vals_diff = out_train[4].mean(axis=0)

        df_out = self.update_eval_csv(
            iter_losses_mean, train, col,
            primal_residuals=primal_residuals,
            dual_residuals=dual_residuals,
            obj_vals_diff=obj_vals_diff
        )
        iters_df, primal_residuals_df, dual_residuals_df, obj_vals_diff_df = df_out

        if not self.skip_startup:
            # write accuracies dataframe to csv
            self.write_accuracies_csv(iter_losses_mean, train, col)

        # plot the evaluation iterations
        self.plot_eval_iters(iters_df, primal_residuals_df,
                             dual_residuals_df, plot_pretrain, obj_vals_diff_df, train, col)

        # SRG-type plots
        # r = out_train[2]
        # self.plot_angles(angles, r, train, col)

        # plot the warm-start predictions
        z_all = out_train[2]

        if isinstance(self.l2ws_model, SCSmodel):
            u_all = out_train[6]
        # u_all = out_train[0][3]
        # z_all = out_train[0][0]
        # self.plot_warm_starts(u_all, z_all, train, col)
            z_plot = z_all[:, :, :-1] / z_all[:, :, -1:]
        else:
            z_plot = z_all
        # import pdb
        # pdb.set_trace()
        self.plot_warm_starts(None, z_plot, train, col)

        # plot the alpha coefficients
        # alpha = out_train[0][2]
        # self.plot_alphas(alpha, train, col)

        # custom visualize
        if self.has_custom_visualization:
            # if isinstance(self.l2ws_model, OSQPmodel):
            #     x_primals = z_all[:, :, :self.l2ws_model.n]
            # elif isinstance(self.l2ws_model, SCSmodel):
            #     x_primals = u_all[:, :, :self.l2ws_model.n] / u_all[:, :, -1:]
            # self.custom_visualize(x_primals, train, col)
            self.custom_visualize(z_all, train, col)

        # closed loop control rollouts
        if not train:
            if self.closed_loop_rollout_dict is not None:
                self.run_closed_loop_rollouts(col)

        # solve with scs
        # z0_mat = z_all[:, 0, :]
        # self.solve_scs(z0_mat, train, col)
        # self.solve_scs(z_all, u_all, train, col)
        z0_mat = z_all[:, 0, :]

        if self.solve_c_num > 0:
            if 'solve_c' in dir(self.l2ws_model):
                self.solve_c_helper(z0_mat, train, col)

        if self.save_weights_flag:
            self.save_weights()

        return out_train

    def solve_c_helper(self, z0_mat, train, col):
        """
        calls the self.solve_c method and does housekeeping
        """
        num_tols = len(self.rel_tols)

        # get the q_mat
        if train:
            q_mat = self.l2ws_model.q_mat_train
        else:
            q_mat = self.l2ws_model.q_mat_test

        # different behavior for prev_sol
        if col == 'prev_sol':
            non_first_indices = jnp.mod(jnp.arange(q_mat.shape[0]), self.traj_length) != 0
            q_mat = q_mat[non_first_indices, :]

        mean_solve_times = np.zeros(num_tols)
        mean_solve_iters = np.zeros(num_tols)
        for i in range(num_tols):
            rel_tol = self.rel_tols[i]
            abs_tol = self.abs_tols[i]
            acc_string = f"abs_{abs_tol}_rel_{rel_tol}"

            solve_c_out = self.l2ws_model.solve_c(
                z0_mat[:self.solve_c_num, :], q_mat[:self.solve_c_num, :], rel_tol, abs_tol)
            solve_times, solve_iters = solve_c_out[0], solve_c_out[1]
            mean_solve_times[i] = solve_times.mean()
            mean_solve_iters[i] = solve_iters.mean()

            # write the solve times to the csv file
            solve_times_df = pd.DataFrame()
            solve_times_df['solve_times'] = solve_times
            solve_times_df['solve_iters'] = solve_iters

            if not os.path.exists('solve_C'):
                os.mkdir('solve_C')
            if train:
                solve_times_path = 'solve_C/train'
            else:
                solve_times_path = 'solve_C/test'
            if not os.path.exists(solve_times_path):
                os.mkdir(solve_times_path)
            if not os.path.exists(f"{solve_times_path}/{col}"):
                os.mkdir(f"{solve_times_path}/{col}")
            if not os.path.exists(solve_times_path):
                os.mkdir(solve_times_path)
            if not os.path.exists(f"{solve_times_path}/{col}/{acc_string}"):
                os.mkdir(f"{solve_times_path}/{col}/{acc_string}")

            solve_times_df.to_csv(f"{solve_times_path}/{col}/{acc_string}/solve_times.csv")

        # update the mean values
        if train:
            train_str = 'train'
            self.agg_solve_times_df_train[col] = mean_solve_times
            self.agg_solve_iters_df_train[col] = mean_solve_iters
            self.agg_solve_times_df_train.to_csv(f"solve_C/{train_str}_aggregate_solve_times.csv")
            self.agg_solve_iters_df_train.to_csv(f"solve_C/{train_str}_aggregate_solve_iters.csv")
        else:
            train_str = 'test'
            self.agg_solve_times_df_test[col] = mean_solve_times
            self.agg_solve_iters_df_test[col] = mean_solve_iters
            self.agg_solve_times_df_test.to_csv(f"solve_C/{train_str}_aggregate_solve_times.csv")
            self.agg_solve_iters_df_test.to_csv(f"solve_C/{train_str}_aggregate_solve_iters.csv")

    def solve_scs(self, z0_mat, train, col):
        # create the path
        if train:
            scs_path = 'scs_train'
        else:
            scs_path = 'scs_test'
        if not os.path.exists(scs_path):
            os.mkdir(scs_path)
        if not os.path.exists(f"{scs_path}/{col}"):
            os.mkdir(f"{scs_path}/{col}")

        # assume M doesn't change across problems
        # extract P, A
        m, n = self.l2ws_model.m, self.l2ws_model.n
        P = csc_matrix(self.l2ws_model.static_M[:n, :n])
        A = csc_matrix(-self.l2ws_model.static_M[n:, :n])

        # set the solver
        b_zeros, c_zeros = np.zeros(m), np.zeros(n)
        scs_data = dict(P=P, A=A, b=b_zeros, c=c_zeros)
        cones_dict = self.cones

        solver = scs.SCS(scs_data,
                         cones_dict,
                         normalize=False,
                         scale=1,
                         adaptive_scale=False,
                         rho_x=1,
                         alpha=1,
                         acceleration_lookback=0,
                         eps_abs=1e-2,
                         eps_rel=1e-2)

        num = 20
        solve_times = np.zeros(num)
        solve_iters = np.zeros(num)

        if train:
            q_mat = self.l2ws_model.q_mat_train
        else:
            q_mat = self.l2ws_model.q_mat_test

        for i in range(num):
            # get the current q
            # q = q_mat[i, :]

            # set b, c
            b, c = q_mat[i, n:], q_mat[i, :n]
            # scs_data = dict(P=P, A=A, b=b, c=c)
            solver.update(b=np.array(b))
            solver.update(c=np.array(c))

            # set the warm start
            x, y, s = self.get_xys_from_z(z0_mat[i, :])

            # solve
            sol = solver.solve(warm_start=True, x=np.array(x), y=np.array(y), s=np.array(s))

            # set the solve time in seconds
            solve_times[i] = sol['info']['solve_time'] / 1000
            solve_iters[i] = sol['info']['iter']
        mean_time = solve_times.mean()
        mean_iters = solve_iters.mean()

        # write the solve times to the csv file
        scs_df = pd.DataFrame()
        scs_df['solve_times'] = solve_times
        scs_df['solve_iters'] = solve_iters
        scs_df['mean_time'] = mean_time
        scs_df['mean_iters'] = mean_iters
        scs_df.to_csv(f"{scs_path}/{col}/scs_solve_times.csv")

    def get_xys_from_z(self, z_init):
        """
        z = (x, y + s, 1)
        we always set the last entry of z to be 1
        we allow s to be zero (we just need z[n:m + n] = y + s)
        """
        m, n = self.l2ws_model.m, self.l2ws_model.n
        x = z_init[:n]
        y = z_init[n:n + m]
        s = jnp.zeros(m)
        return x, y, s
    
    def run_closed_loop_rollouts(self, col):
        """
        implements the closed_loop_rollouts

        qp_solver will depend on the col
        - if cold-start or trained: run through neural network
        - if nearest-neighbor, compute nn on-the-fly
        - if prev-sol, need function to get previous sol
        """
        num_rollouts = self.closed_loop_rollout_dict['num_rollouts']
        rollout_length = self.closed_loop_rollout_dict['rollout_length']
        dynamics = self.closed_loop_rollout_dict['dynamics']
        u_init_traj = self.closed_loop_rollout_dict['u_init_traj']
        system_constants = self.closed_loop_rollout_dict['system_constants']
        plot_traj = self.closed_loop_rollout_dict.get('plot_traj', None)
        # ref_traj_dict_lists = self.closed_loop_rollout_dict['ref_traj_dict_lists_test']
        ref_traj_tensor = self.closed_loop_rollout_dict['ref_traj_tensor']
        budget = self.closed_loop_rollout_dict['closed_loop_budget']
        dt, nx = system_constants['dt'], system_constants['nx']
        cd0, T = system_constants['cd0'], system_constants['T']

        Q_ref =  self.closed_loop_rollout_dict['Q_ref']
        obstacle_tol = self.closed_loop_rollout_dict['obstacle_tol']

        static_canon_mpc_osqp_partial = self.closed_loop_rollout_dict['static_canon_mpc_osqp_partial']

        # setup the qp_solver
        qp_solver = partial(self.qp_solver, dt=dt, cd0=cd0, nx=nx, method=col, static_canon_mpc_osqp_partial=static_canon_mpc_osqp_partial)

        num_goals = ref_traj_tensor.shape[1]
        N_train = self.thetas_train.shape[0]
        # num_train_rollouts = int(N_train / (rollout_length - T))
        num_train_rollouts = int(N_train / (rollout_length))

        # do the closed loop rollouts
        rollout_results_list = []
        for i in range(num_rollouts):
            # get x_init_traj
            thetas_index = i * rollout_length
            x_init_traj = self.thetas_test[thetas_index, :nx] # assumes theta = (x0, u0, x_ref)
            print('x_init_traj', x_init_traj)

            # old
            # ref_traj_index = num_train_rollouts + i
            # traj_list = [ref_traj_tensor[ref_traj_index, i, :] for i in range(num_goals)]
            # ref_traj_dict = dict(case='obstacle_course', traj_list=traj_list, Q=Q_ref, tol=obstacle_tol)
            ref_traj_index = num_train_rollouts + i
            trajectories = ref_traj_tensor[ref_traj_index, :, :]
            ref_traj_dict = dict(case='loop_path', traj_list=trajectories, Q=Q_ref, tol=obstacle_tol)

            # new
            rollout_results = closed_loop_rollout(qp_solver,
                                rollout_length,
                                x_init_traj,
                                u_init_traj,
                                dynamics,
                                system_constants,
                                ref_traj_dict,
                                budget,
                                noise_list=None)
            rollout_results_list.append(rollout_results)
            state_traj_list = rollout_results['state_traj_list']

            # plot and save the rollout results
            if not os.path.exists('rollouts'):
                os.mkdir('rollouts')
            if not os.path.exists(f"rollouts/{col}"):
                os.mkdir(f"rollouts/{col}")
            traj_list = ref_traj_dict['traj_list']
            
            if plot_traj is not None:
                plot_traj([state_traj_list], goals=traj_list, labels=[col], filename=f"rollouts/{col}/rollout_{i}")

    def qp_solver(self, Ac, Bc, x0, u0, x_dot, ref_traj, budget, prev_sol, dt, cd0, nx, static_canon_mpc_osqp_partial, method):
        """
        method could be one of the following
        - cold-start
        - nearest-neighbor
        - prev-sol
        - anything learned
        """
        # get the discrete time system Ad, Bd from the continuous time system Ac, Bc
        Ad = jnp.eye(nx) + Ac * dt
        Bd = Bc * dt
        # print('Bd', Bd)
        # no need to use u0 for the non-learned case

        # get the constants for the discrete system
        cd = cd0 + (x_dot - Ac @ x0 - Bc @ u0) * dt

        # get (P, A, c, l, u)
        out_dict = static_canon_mpc_osqp_partial(ref_traj, x0, Ad, Bd, cd=cd, u_prev=u0)
        P, A, c, l, u = out_dict['P'], out_dict['A'], out_dict['c'], out_dict['l'], out_dict['u']
        m, n = A.shape
        q = jnp.concatenate([c, l, u])
        # print('q', q[:30])

        # get factor
        rho_vec, sigma = jnp.ones(m), 1
        rho_vec = rho_vec.at[l == u].set(1000)
        M = P + sigma * jnp.eye(n) + A.T @ jnp.diag(rho_vec) @ A
        factor = jsp.linalg.lu_factor(M)

        # solve
        # z0 = prev_sol  # jnp.zeros(m + n)
        # out = k_steps_eval_osqp(budget, z0, q, factor, P, A, rho=rho_vec,
        #                         sigma=sigma, supervised=False, z_star=None, jit=True)

        # expand so that vectors become matrices
        #   i.e. we are only feeding one input into our method, but out method handles batches
        #   input has shape (d), but inputs has shape (1, d)
        factors = (jnp.expand_dims(factor[0], 0), jnp.expand_dims(factor[1], 0))

        q_full = jnp.concatenate([q, vec_symm(P), jnp.reshape(A, (m * n))])
        q_mat = jnp.expand_dims(q_full, 0)
        z_stars = None

        # get theta
        # theta = jnp.concatenate([x0, u0, ref_traj[:3]]) # assumes specific form of theta
        theta = jnp.concatenate([x0, u0, jnp.ravel(ref_traj[:, :3])])
        print('theta', theta)

        # need to transform the input
        if method == 'nearest_neighbor':
            inputs = self.theta_2_nearest_neighbor(theta)
            fixed_ws = True
        elif method == 'prev_sol':
            # input = self.shifted_sol(prev_sol)
            prev_sol_mat = jnp.expand_dims(prev_sol, 0)
            inputs = self.shifted_sol_fn(prev_sol_mat)
            # inputs = jnp.expand_dims(input, 0)
            fixed_ws = True
        else:
            normalized_input = self.normalize_theta(theta)
            inputs = jnp.expand_dims(normalized_input, 0)
            fixed_ws = False
            print('inputs', inputs)
            # import pdb
            # pdb.set_trace()

        loss, out, time_per_prob  = self.l2ws_model.dynamic_eval(budget, inputs, q_mat, z_stars, factors, tag='test', fixed_ws=fixed_ws)
        

        # sol = out[0]
        sol = out[2][0, -1, :]
        print('loss', out[1][-1])
        # plt.plot(out[1])
        # plt.yscale('log')
        # plt.show()
        # plt.clf()

        # z0 = sol[:nx]
        # w0 = sol[T*nx:T*nx + nu]
        # z1 = sol[nx:2*nx]
        # w1 = sol[T*nx + nu:T*nx + 2*nu]

        return sol, P, A, factor, q
    

    # def custom_visualize(self, x_primals, train, col):
    #     """
    #     x_primals has shape [N, eval_iters]
    #     """
    #     visualize_path = 'visualize_train' if train else 'visualize_test'

    #     if not os.path.exists(visualize_path):
    #         os.mkdir(visualize_path)
    #     if not os.path.exists(f"{visualize_path}/{col}"):
    #         os.mkdir(f"{visualize_path}/{col}")

    #     visual_path = f"{visualize_path}/{col}"

    #     # call custom visualize fn
    #     if train:
    #         # x_stars = self.l2ws_model.x_stars_train
    #         x_stars = self.x_stars_train
    #         thetas = self.thetas_train
    #         if 'x_nn_train' in dir(self):
    #             x_nn = self.x_nn_train
    #     else:
    #         # x_stars = self.l2ws_model.x_stars_test
    #         x_stars = self.x_stars_test
    #         thetas = self.thetas_test
    #         if 'x_nn_test' in dir(self):
    #             x_nn = self.x_nn_test

    #     if col == 'no_train':
    #         if train:
    #             self.x_no_learn_train = x_primals[:self.vis_num, :, :]
    #         else:
    #             self.x_no_learn_test = x_primals[:self.vis_num, :, :]
    #     elif col == 'nearest_neighbor':
    #         if train:
    #             self.x_nn_train = x_primals[:self.vis_num, :, :]
    #         else:
    #             self.x_nn_test = x_primals[:self.vis_num, :, :]
    #     if train:
    #         x_no_learn = self.x_no_learn_train[:self.vis_num, :, :]
    #     else:
    #         x_no_learn = self.x_no_learn_test[:self.vis_num, :, :]

    #     if col != 'nearest_neighbor' and col != 'no_train':
    #         self.custom_visualize_fn(x_primals, x_stars, x_no_learn, x_nn,
    #                                  thetas, self.iterates_visualize, visual_path)


    def custom_visualize(self, z_all, train, col):
        """
        x_primals has shape [N, eval_iters]
        """
        visualize_path = 'visualize_train' if train else 'visualize_test'

        if not os.path.exists(visualize_path):
            os.mkdir(visualize_path)
        if not os.path.exists(f"{visualize_path}/{col}"):
            os.mkdir(f"{visualize_path}/{col}")

        visual_path = f"{visualize_path}/{col}"

        # call custom visualize fn
        if train:
            z_stars = self.z_stars_train
            thetas = self.thetas_train
            if 'z_nn_train' in dir(self):
                z_nn = self.z_nn_train
        else:
            z_stars = self.z_stars_test
            thetas = self.thetas_test
            if 'z_nn_test' in dir(self):
                z_nn = self.z_nn_test

        if col == 'no_train':
            if train:
                self.z_no_learn_train = z_all #[:self.vis_num, :, :]
            else:
                self.z_no_learn_test = z_all # x_primals[:self.vis_num, :, :]
        elif col == 'nearest_neighbor':
            if train:
                self.z_nn_train = z_all #x_primals[:self.vis_num, :, :]
            else:
                self.z_nn_test = z_all #x_primals[:self.vis_num, :, :]
        if train:
            z_no_learn = self.z_no_learn_train
        else:
            z_no_learn = self.z_no_learn_test

        if col != 'nearest_neighbor' and col != 'no_train':
            self.custom_visualize_fn(z_all, z_stars, z_no_learn, z_nn,
                                     thetas, self.iterates_visualize, visual_path)


    def run(self):
        # setup logging and dataframes
        self._init_logging()
        self.setup_dataframes()

        # set pretrain_on boolean
        self.pretrain_on = self.pretrain_cfg.pretrain_iters > 0

        if not self.skip_startup:
            # no learning evaluation
            self.eval_iters_train_and_test('no_train', False)

            # fixed ws evaluation
            if self.l2ws_model.z_stars_train is not None:
                self.eval_iters_train_and_test('nearest_neighbor', False)

            # prev sol eval
            if self.prev_sol_eval and self.l2ws_model.z_stars_train is not None:
                self.eval_iters_train_and_test('prev_sol', False)

            # pretrain evaluation
            if self.pretrain_on:
                self.pretrain()

        # load the weights AFTER the cold-start
        if self.load_weights_datetime is not None:
            self.load_weights(self.example, self.load_weights_datetime)

        # eval test data to start
        self.test_eval_write()

        # do all of the training
        test_zero = True if self.skip_startup else False
        self.train(test_zero=test_zero)

    def train(self, test_zero=False):
        """
        does all of the training
        jits together self.epochs_jit number of epochs together
        writes results to filesystem
        """
        num_epochs_jit = int(self.l2ws_model.epochs / self.epochs_jit)
        loop_size = int(self.l2ws_model.num_batches * self.epochs_jit)

        # key_count updated to get random permutation for each epoch
        # key_count = 0

        for epoch_batch in range(num_epochs_jit):
            epoch = int(epoch_batch * self.epochs_jit)
            if test_zero or epoch % self.eval_every_x_epochs == 0 and epoch > 0:
                self.eval_iters_train_and_test(f"train_epoch_{epoch}", self.pretrain_on)
            # if epoch > self.l2ws_model.dont_decay_until:
            #     self.l2ws_model.decay_upon_plateau()

            # setup the permutations
            permutation = setup_permutation(
                self.key_count, self.l2ws_model.N_train, self.epochs_jit)

            # train the jitted epochs
            params, state, epoch_train_losses, time_train_per_epoch = self.train_jitted_epochs(
                permutation, epoch)

            # reset the global (params, state)
            self.key_count += 1
            self.l2ws_model.epoch += self.epochs_jit
            self.l2ws_model.params, self.l2ws_model.state = params, state

            prev_batches = len(self.l2ws_model.tr_losses_batch)
            self.l2ws_model.tr_losses_batch = self.l2ws_model.tr_losses_batch + \
                list(epoch_train_losses)

            # write train results
            self.write_train_results(loop_size, prev_batches,
                                     epoch_train_losses, time_train_per_epoch)

            # evaluate the test set and write results
            self.test_eval_write()

            # plot the train / test loss so far
            if epoch % self.save_every_x_epochs == 0:
                self.plot_train_test_losses()

    def train_jitted_epochs(self, permutation, epoch):
        """
        train self.epochs_jit at a time
        special case: the first time we call train_batch (i.e. epoch = 0)
        """
        epoch_batch_start_time = time.time()
        loop_size = int(self.l2ws_model.num_batches * self.epochs_jit)
        epoch_train_losses = jnp.zeros(loop_size)
        if epoch == 0:
            # unroll the first iterate so that This allows `init_val` and `body_fun`
            #   below to have the same output type, which is a requirement of
            #   lax.while_loop and lax.scan.
            batch_indices = lax.dynamic_slice(
                permutation, (0,), (self.l2ws_model.batch_size,))
            train_loss_first, params, state = self.l2ws_model.train_batch(
                batch_indices, self.l2ws_model.params, self.l2ws_model.state)

            epoch_train_losses = epoch_train_losses.at[0].set(train_loss_first)
            start_index = 1
        else:
            start_index = 0
            params, state = self.l2ws_model.params, self.l2ws_model.state

        # loop the last (self.l2ws_model.num_batches - 1) iterates if not
        #   the first time calling train_batch
        init_val = epoch_train_losses, params, state
        body_fn = partial(self.train_over_epochs_body_fn, permutation=permutation)
        val = lax.fori_loop(start_index, loop_size, body_fn, init_val)

        epoch_batch_end_time = time.time()
        time_diff = epoch_batch_end_time - epoch_batch_start_time
        time_train_per_epoch = time_diff / self.epochs_jit
        epoch_train_losses, params, state = val
        return params, state, epoch_train_losses, time_train_per_epoch

    def train_over_epochs_body_fn(self, batch, val, permutation):
        """
        to be used as the body_fn in lax.fori_loop
        need to call partial for the specific permutation
        """
        train_losses, params, state = val
        start_index = batch * self.l2ws_model.batch_size
        batch_indices = lax.dynamic_slice(
            permutation, (start_index,), (self.l2ws_model.batch_size,))
        train_loss, params, state = self.l2ws_model.train_batch(
            batch_indices, params, state)
        train_losses = train_losses.at[batch].set(train_loss)
        val = train_losses, params, state
        return val

    def write_accuracies_csv(self, losses, train, col):
        df_acc = pd.DataFrame()
        df_acc['accuracies'] = np.array(self.accs)

        if train:
            accs_path = 'accuracies_train'
        else:
            accs_path = 'accuracies_test'
        if not os.path.exists(accs_path):
            os.mkdir(accs_path)
        if not os.path.exists(f"{accs_path}/{col}"):
            os.mkdir(f"{accs_path}/{col}")

        # accuracies
        iter_vals = np.zeros(len(self.accs))
        for i in range(len(self.accs)):
            if losses.min() < self.accs[i]:
                iter_vals[i] = int(np.argmax(losses < self.accs[i]))
            else:
                iter_vals[i] = losses.size
        int_iter_vals = iter_vals.astype(int)
        df_acc[col] = int_iter_vals
        df_acc.to_csv(f"{accs_path}/{col}/accuracies.csv")

        # save no learning accuracies
        if col == 'no_train':
            self.no_learning_accs = int_iter_vals

        # percent reduction
        df_percent = pd.DataFrame()
        df_percent['accuracies'] = np.array(self.accs)

        for col in df_acc.columns:
            if col != 'accuracies':
                val = 1 - df_acc[col] / self.no_learning_accs
                df_percent[col] = np.round(val, decimals=2)
        df_percent.to_csv(f"{accs_path}/{col}/reduction.csv")

    def eval_iters_train_and_test(self, col, pretrain_on):
        self.evaluate_iters(
            self.num_samples, col, train=True, plot_pretrain=pretrain_on)
        self.evaluate_iters(
            self.num_samples, col, train=False, plot_pretrain=pretrain_on)

    def write_train_results(self, loop_size, prev_batches, epoch_train_losses,
                            time_train_per_epoch):
        for batch in range(loop_size):
            start_window = prev_batches - 10 + batch
            end_window = prev_batches + batch
            last10 = np.array(self.l2ws_model.tr_losses_batch[start_window:end_window])
            moving_avg = last10.mean()
            self.writer.writerow({
                'train_loss': epoch_train_losses[batch],
                'moving_avg_train': moving_avg,
                'time_train_per_epoch': time_train_per_epoch
            })
            self.logf.flush()

    def evaluate_only(self, fixed_ws, num, train, col, batch_size):
        tag = 'train' if train else 'test'
        if self.static_flag:
            factors = None
        else:
            if train:
                factors = (self.factors_train[0][:num, :, :], self.factors_train[1][:num, :])
            else:
                factors = (self.factors_test[0][:num, :, :], self.factors_test[1][:num, :])
        if self.l2ws_model.z_stars_train is None:
            z_stars = None
        else:
            z_stars = self.l2ws_model.z_stars_train[:num,
                                                    :] if train else self.l2ws_model.z_stars_test[:num,
                                                                                                  :]
        if col == 'prev_sol':
            q_mat_full = self.l2ws_model.q_mat_train[:num,
                                                     :] if train else self.l2ws_model.q_mat_test[:num, :]
            non_first_indices = jnp.mod(jnp.arange(num), self.traj_length) != 0
            q_mat = q_mat_full[non_first_indices, :]
            z_stars = z_stars[non_first_indices, :]
            factors = (factors[0][non_first_indices, :, :], factors[1][non_first_indices, :])
        else:
            q_mat = self.l2ws_model.q_mat_train[:num,
                                                :] if train else self.l2ws_model.q_mat_test[:num, :]

        inputs = self.get_inputs_for_eval(fixed_ws, num, train, col)

        # do the batching
        num_batches = int(num / batch_size)
        full_eval_out = []
        if num_batches == 1:
            eval_out = self.l2ws_model.evaluate(
                self.eval_unrolls, inputs, q_mat, z_stars, fixed_ws, factors=factors, tag=tag)
            return eval_out

        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            curr_inputs = inputs[start: end]
            curr_q_mat = q_mat[start: end]
            if z_stars is not None:
                curr_z_stars = z_stars[start: end]
            else:
                curr_z_stars = None
            eval_out = self.l2ws_model.evaluate(
                self.eval_unrolls, curr_inputs, curr_q_mat, curr_z_stars, fixed_ws, factors=factors, tag=tag)
            full_eval_out.append(eval_out)
        loss = np.array([curr_out[0] for curr_out in full_eval_out]).mean()
        time_per_prob = np.array([curr_out[2] for curr_out in full_eval_out]).mean()
        out = self.stack_tuples([curr_out[1] for curr_out in full_eval_out])

        flattened_eval_out = (loss, out, time_per_prob)
        return flattened_eval_out

    def stack_tuples(self, tuples_list):
        result = []
        num_tuples = len(tuples_list)
        tuple_length = len(tuples_list[0])

        for i in range(tuple_length):
            stacked_entry = []
            for j in range(num_tuples):
                stacked_entry.append(tuples_list[j][i])
            # result.append(tuple(stacked_entry))
            if tuples_list[j][i].ndim == 2:
                result.append(jnp.vstack(stacked_entry))
            elif tuples_list[j][i].ndim == 1:
                result.append(jnp.hstack(stacked_entry))
            elif tuples_list[j][i].ndim == 3 and i == 0:
                result.append(jnp.vstack(stacked_entry))
        return result

    def get_inputs_for_eval(self, fixed_ws, num, train, col):
        if fixed_ws:
            if col == 'nearest_neighbor':
                inputs = self.get_nearest_neighbors(train, num)
            elif col == 'prev_sol':
                # z_size = self.z_stars_test.shape[1]
                # inputs = jnp.zeros((num, z_size))
                # inputs = inputs.at[1:, :].set(self.z_stars_test[:num - 1, :])

                # now set the indices (0, num_traj, 2 * num_traj) to zero
                non_last_indices = jnp.mod(jnp.arange(
                    num), self.traj_length) != self.traj_length - 1
                # inputs = inputs[non_last_indices, :]
                # inputs = self.shifted_sol_fn(inputs)

                inputs = self.shifted_sol_fn(self.z_stars_test[:num, :][non_last_indices, :])
        else:
            if train:
                inputs = self.l2ws_model.train_inputs[:num, :]
            else:
                inputs = self.l2ws_model.test_inputs[:num, :]
        return inputs
    
    def theta_2_nearest_neighbor(self, theta):
        """
        given a new theta returns the closest training problem solution
        """
        # first normalize theta
        test_input = self.normalize_theta(theta)

        # make it a matrix
        test_inputs = jnp.expand_dims(test_input, 0)

        distances = distance_matrix(
                np.array(test_inputs),
                np.array(self.l2ws_model.train_inputs))
        indices = np.argmin(distances, axis=1)
        if isinstance(self.l2ws_model, OSQPmodel):
            return self.l2ws_model.z_stars_train[indices, :self.m + self.n]
        else:
            return self.l2ws_model.z_stars_train[indices, :]

    def get_nearest_neighbors(self, train, num):
        if train:
            distances = distance_matrix(
                np.array(self.l2ws_model.train_inputs[:num, :]),
                np.array(self.l2ws_model.train_inputs))
        else:
            distances = distance_matrix(
                np.array(self.l2ws_model.test_inputs[:num, :]),
                np.array(self.l2ws_model.train_inputs))
        print('distances', distances)
        indices = np.argmin(distances, axis=1)
        print('indices', indices)
        best_val = np.min(distances, axis=1)
        print('best val', best_val)
        plt.plot(indices)
        if train:
            plt.savefig("indices_train_plot.pdf", bbox_inches='tight')
        else:
            plt.savefig("indices_train_plot.pdf", bbox_inches='tight')
        plt.clf()
        if isinstance(self.l2ws_model, OSQPmodel):
            return self.l2ws_model.z_stars_train[indices, :self.m + self.n]
        return self.l2ws_model.z_stars_train[indices, :]

    def setup_dataframes(self):
        self.iters_df_train = pd.DataFrame(
            columns=['iterations', 'no_train'])
        self.iters_df_train['iterations'] = np.arange(1, self.eval_unrolls+1)

        self.iters_df_test = pd.DataFrame(
            columns=['iterations', 'no_train'])
        self.iters_df_test['iterations'] = np.arange(1, self.eval_unrolls+1)

        # primal and dual residuals
        self.primal_residuals_df_train = pd.DataFrame(
            columns=['iterations'])
        self.primal_residuals_df_train['iterations'] = np.arange(
            1, self.eval_unrolls+1)
        self.dual_residuals_df_train = pd.DataFrame(
            columns=['iterations'])
        self.dual_residuals_df_train['iterations'] = np.arange(
            1, self.eval_unrolls+1)

        self.primal_residuals_df_test = pd.DataFrame(
            columns=['iterations'])
        self.primal_residuals_df_test['iterations'] = np.arange(
            1, self.eval_unrolls+1)
        self.dual_residuals_df_test = pd.DataFrame(
            columns=['iterations'])
        self.dual_residuals_df_test['iterations'] = np.arange(
            1, self.eval_unrolls+1)

        # obj_vals_diff
        self.obj_vals_diff_df_train = pd.DataFrame(
            columns=['iterations'])
        self.obj_vals_diff_df_train['iterations'] = np.arange(
            1, self.eval_unrolls+1)
        self.obj_vals_diff_df_test = pd.DataFrame(
            columns=['iterations'])
        self.obj_vals_diff_df_test['iterations'] = np.arange(
            1, self.eval_unrolls+1)

        # setup solve times
        self.agg_solve_times_df_train = pd.DataFrame()
        self.agg_solve_times_df_train['rel_tol'] = self.rel_tols
        self.agg_solve_times_df_train['abs_tol'] = self.abs_tols
        self.agg_solve_iters_df_train = pd.DataFrame()
        self.agg_solve_iters_df_train['rel_tol'] = self.rel_tols
        self.agg_solve_iters_df_train['abs_tol'] = self.abs_tols

        self.agg_solve_times_df_test = pd.DataFrame()
        self.agg_solve_times_df_test['rel_tol'] = self.rel_tols
        self.agg_solve_times_df_test['abs_tol'] = self.abs_tols
        self.agg_solve_iters_df_test = pd.DataFrame()
        self.agg_solve_iters_df_test['rel_tol'] = self.rel_tols
        self.agg_solve_iters_df_test['abs_tol'] = self.abs_tols

    def train_full(self):
        print("Training full...")
        pretrain_on = True
        self.full_train_df = pd.DataFrame(
            columns=['pretrain_loss', 'pretrain_test_loss'])
        pretrain_out = self.l2ws_model.train_full(self.l2ws_model.static_algo_factor,
                                                  self.proj,
                                                  self.train_unrolls,
                                                  1000,
                                                  stepsize=1e-4,
                                                  df_fulltrain=self.full_train_df,
                                                  batches=10)
        train_pretrain_losses, test_pretrain_losses = pretrain_out
        self.evaluate_iters(
            self.num_samples, 'pretrain', train=True, plot_pretrain=pretrain_on)
        self.evaluate_iters(
            self.num_samples, 'pretrain', train=False, plot_pretrain=pretrain_on)
        plt.plot(train_pretrain_losses, label='train')
        plt.plot(test_pretrain_losses, label='test')
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.ylabel('full train loss')
        plt.legend()
        plt.savefig('losses.pdf')
        plt.clf()

    def pretrain(self):
        print("Pretraining...")
        pretrain_on = True
        self.df_pretrain = pd.DataFrame(
            columns=['pretrain_loss', 'pretrain_test_loss'])
        pretrain_out = self.l2ws_model.pretrain(self.pretrain_cfg.pretrain_iters,
                                                stepsize=self.pretrain_cfg.pretrain_stepsize,
                                                df_pretrain=self.df_pretrain,
                                                batches=self.pretrain_cfg.pretrain_batches)
        train_pretrain_losses, test_pretrain_losses = pretrain_out
        self.evaluate_iters(
            self.num_samples, 'pretrain', train=True, plot_pretrain=pretrain_on)
        self.evaluate_iters(
            self.num_samples, 'pretrain', train=False, plot_pretrain=pretrain_on)
        plt.plot(train_pretrain_losses, label='train')
        plt.plot(test_pretrain_losses, label='test')
        plt.yscale('log')
        plt.xlabel('pretrain iterations')
        plt.ylabel('pretrain loss')
        plt.legend()
        plt.savefig('pretrain_losses.pdf')
        plt.clf()

    def test_eval_write(self):
        test_loss, time_per_iter = self.l2ws_model.short_test_eval()
        # test_loss, time_per_iter = 1, 1
        last_epoch = np.array(self.l2ws_model.tr_losses_batch[-self.l2ws_model.num_batches:])
        moving_avg = last_epoch.mean()
        if self.test_writer is not None:
            self.test_writer.writerow({
                'iter': self.l2ws_model.state.iter_num,
                'train_loss': moving_avg,
                'test_loss': test_loss,
                'time_per_iter': time_per_iter
            })
            self.test_logf.flush()

    def plot_train_test_losses(self):
        batch_losses = np.array(self.l2ws_model.tr_losses_batch)
        te_losses = np.array(self.l2ws_model.te_losses)
        num_data_points = batch_losses.size
        epoch_axis = np.arange(num_data_points) / \
            self.l2ws_model.num_batches

        epoch_test_axis = self.epochs_jit * np.arange(te_losses.size)
        plt.plot(epoch_axis, batch_losses, label='train')
        plt.plot(epoch_test_axis, te_losses, label='test')
        plt.yscale('log')
        plt.xlabel('epochs')
        plt.ylabel('fixed point residual average')
        plt.legend()
        plt.savefig('losses_over_training.pdf', bbox_inches='tight')
        plt.clf()

        plt.plot(epoch_axis, batch_losses, label='train')

        # include when learning rate decays
        if len(self.l2ws_model.epoch_decay_points) > 0:
            epoch_decay_points = self.l2ws_model.epoch_decay_points
            epoch_decay_points_np = np.array(epoch_decay_points)
            batch_decay_points = epoch_decay_points_np * self.l2ws_model.num_batches

            batch_decay_points_int = batch_decay_points.astype('int')
            decay_vals = batch_losses[batch_decay_points_int]
            plt.scatter(epoch_decay_points_np, decay_vals, c='r', label='lr decay')
        plt.yscale('log')
        plt.xlabel('epochs')
        plt.ylabel('fixed point residual average')
        plt.legend()
        plt.savefig('train_losses_over_training.pdf', bbox_inches='tight')
        plt.clf()

    def update_eval_csv(self, iter_losses_mean, train, col, primal_residuals=None,
                        dual_residuals=None, obj_vals_diff=None):
        # def update_eval_csv(self, iter_losses_mean, train, col):
        """
        update the eval csv files
            fixed point residuals
            primal residuals
            dual residuals
        returns the new dataframes
        """
        primal_residuals_df, dual_residuals_df = None, None
        obj_vals_diff_df = None
        if train:
            self.iters_df_train[col] = iter_losses_mean
            self.iters_df_train.to_csv('iters_compared_train.csv')
            if primal_residuals is not None:
                self.primal_residuals_df_train[col] = primal_residuals
                self.primal_residuals_df_train.to_csv('primal_residuals_train.csv')
                self.dual_residuals_df_train[col] = dual_residuals
                self.dual_residuals_df_train.to_csv('dual_residuals_train.csv')
                primal_residuals_df = self.primal_residuals_df_train
                dual_residuals_df = self.dual_residuals_df_train
            if obj_vals_diff is not None:
                self.obj_vals_diff_df_train[col] = obj_vals_diff
                self.obj_vals_diff_df_train.to_csv('obj_vals_diff_train.csv')
                obj_vals_diff_df = self.obj_vals_diff_df_train
            iters_df = self.iters_df_train

        else:
            self.iters_df_test[col] = iter_losses_mean
            self.iters_df_test.to_csv('iters_compared_test.csv')
            if primal_residuals is not None:
                self.primal_residuals_df_test[col] = primal_residuals
                self.primal_residuals_df_test.to_csv('primal_residuals_test.csv')
                self.dual_residuals_df_test[col] = dual_residuals
                self.dual_residuals_df_test.to_csv('dual_residuals_test.csv')
                primal_residuals_df = self.primal_residuals_df_test
                dual_residuals_df = self.dual_residuals_df_test
            if obj_vals_diff is not None:
                self.obj_vals_diff_df_test[col] = obj_vals_diff
                self.obj_vals_diff_df_test.to_csv('obj_vals_diff_test.csv')
                obj_vals_diff_df = self.obj_vals_diff_df_test

            iters_df = self.iters_df_test

        return iters_df, primal_residuals_df, dual_residuals_df, obj_vals_diff_df

    def plot_eval_iters_df(self, df, train, col, ylabel, filename):
        # plot the cold-start if applicable
        if 'no_train' in df.keys():
            plt.plot(df['no_train'], 'k-', label='no learning')

        # plot the nearest_neighbor if applicable
        if col != 'no_train' and 'nearest_neighbor' in df.keys():
            plt.plot(df['nearest_neighbor'], 'm-', label='nearest neighbor')

        # plot the prev_sol if applicable
        if col != 'no_train' and col != 'nearest_neighbor' and 'prev_sol' in df.keys():
            plt.plot(df['prev_sol'], 'c-', label='prev solution')
        # if plot_pretrain:
        #     plt.plot(iters_df['pretrain'], 'r-', label='pretraining')

        # plot the learned warm-start if applicable
        if col != 'no_train' and col != 'pretrain' and col != 'nearest_neighbor' and col != 'prev_sol':
            plt.plot(df[col], label=f"train k={self.train_unrolls}")
        plt.yscale('log')
        plt.xlabel('evaluation iterations')
        plt.ylabel(f"test {ylabel}")
        plt.legend()
        if train:
            plt.title('train problems')
            plt.savefig(f"{filename}_train.pdf", bbox_inches='tight')
        else:
            plt.title('test problems')
            plt.savefig(f"{filename}_test.pdf", bbox_inches='tight')
        plt.clf()

    def plot_eval_iters(self, iters_df, primal_residuals_df, dual_residuals_df, plot_pretrain,
                        obj_vals_diff_df,
                        train, col):
        self.plot_eval_iters_df(iters_df, train, col, 'fixed point residual', 'eval_iters')
        if primal_residuals_df is not None:
            self.plot_eval_iters_df(primal_residuals_df, train, col,
                                    'primal residual', 'primal_residuals')
            self.plot_eval_iters_df(dual_residuals_df, train, col,
                                    'dual residual', 'dual_residuals')
        if obj_vals_diff_df is not None:
            self.plot_eval_iters_df(obj_vals_diff_df, train, col, 'obj diff', 'obj_diffs')

    # def plot_eval_iters(self, iters_df, primal_residuals_df, dual_residuals_df, plot_pretrain,
    #                     train, col):
    #     # plot the cold-start if applicable
    #     if 'no_train' in iters_df.keys():
    #         plt.plot(iters_df['no_train'], 'k-', label='no learning')

    #     # plot the nearest_neighbor if applicable
    #     if col != 'no_train' and 'nearest_neighbor' in iters_df.keys():
    #         plt.plot(iters_df['nearest_neighbor'], 'm-', label='nearest neighbor')

    #     # plot the prev_sol if applicable
    #     if col != 'no_train' and col != 'nearest_neighbor' and 'prev_sol' in iters_df.keys():
    #         plt.plot(iters_df['prev_sol'], 'c-', label='prev solution')
    #     # if plot_pretrain:
    #     #     plt.plot(iters_df['pretrain'], 'r-', label='pretraining')

    #     # plot the learned warm-start if applicable
    #     if col != 'no_train' and col != 'pretrain' and col != 'nearest_neighbor' and col != 'prev_sol':
    #         plt.plot(iters_df[col], label=f"train k={self.train_unrolls}")
    #     plt.yscale('log')
    #     plt.xlabel('evaluation iterations')
    #     plt.ylabel('test fixed point residuals')
    #     plt.legend()
    #     if train:
    #         plt.title('train problems')
    #         plt.savefig('eval_iters_train.pdf', bbox_inches='tight')
    #     else:
    #         plt.title('test problems')
    #         plt.savefig('eval_iters_test.pdf', bbox_inches='tight')
    #     plt.clf()

    #     # plot of the primal residuals
    #     if 'no_train' in iters_df.keys():
    #         plt.plot(primal_residuals_df['no_train'],
    #                 'k-', label='no learning primal')
    #         # plt.plot(dual_residuals_df['no_train'],
    #         #         'ko', label='no learning dual')

    #     # plot the nearest_neighbor if applicable
    #     if col != 'no_train' and 'nearest_neighbor' in iters_df.keys():
    #         plt.plot(primal_residuals_df['nearest_neighbor'], 'm-', label='nearest neighbor')

    #     # plot the prev_sol if applicable
    #     if col != 'no_train' and col != 'nearest_neighbor' and 'prev_sol' in iters_df.keys():
    #         plt.plot(primal_residuals_df['prev_sol'], 'c-', label='prev solution')

    #     if col != 'no_train' and col != 'pretrain' and col != 'fixed_ws' and col != 'prev_sol':
    #         plt.plot(
    #             primal_residuals_df[col], label=f"train k={self.train_unrolls} primal")
    #         # plt.plot(
    #         #     dual_residuals_df[col], label=f"train k={self.train_unrolls} dual")
    #     plt.yscale('log')
    #     plt.xlabel('evaluation iterations')
    #     plt.ylabel('primal residuals')
    #     plt.legend()
    #     # plt.savefig('primal_residuals.pdf', bbox_inches='tight')
    #     # plt.clf()
    #     if train:
    #         plt.title('train problems')
    #         plt.savefig('primal_residuals_train.pdf', bbox_inches='tight')
    #     else:
    #         plt.title('test problems')
    #         plt.savefig('primal_residuals_test.pdf', bbox_inches='tight')
    #     plt.clf()

    #     # plot of the dual residuals
    #     if 'no_train' in iters_df.keys():
    #         # plt.plot(primal_residuals_df['no_train'],
    #         #         'k+', label='no learning primal')
    #         plt.plot(dual_residuals_df['no_train'],
    #                 'k-', label='no learning dual')

    #     # plot the nearest_neighbor if applicable
    #     if col != 'no_train' and 'nearest_neighbor' in iters_df.keys():
    #         plt.plot(dual_residuals_df['nearest_neighbor'], 'm-', label='nearest neighbor')

    #     # plot the prev_sol if applicable
    #     if col != 'no_train' and col != 'nearest_neighbor' and 'prev_sol' in iters_df.keys():
    #         plt.plot(dual_residuals_df['prev_sol'], 'c-', label='prev solution')

    #     if col != 'no_train' and col != 'pretrain' and col != 'fixed_ws' and col != 'prev_sol':
    #         # plt.plot(
    #         #     primal_residuals_df[col], label=f"train k={self.train_unrolls} primal")
    #         plt.plot(
    #             dual_residuals_df[col], label=f"train k={self.train_unrolls} dual")
    #     plt.yscale('log')
    #     plt.xlabel('evaluation iterations')
    #     plt.ylabel('dual residuals')
    #     plt.legend()
    #     # plt.savefig('dual_residuals.pdf', bbox_inches='tight')
    #     if train:
    #         plt.title('train problems')
        #     plt.savefig('dual_residuals_train.pdf', bbox_inches='tight')
        # else:
        #     plt.title('test problems')
        #     plt.savefig('dual_residuals_test.pdf', bbox_inches='tight')
        # plt.clf()

    def plot_alphas(self, alpha, train, col):
        """
        in the shared solution method, this plots the alpha coefficients predicted
        """
        if train:
            alpha_path = 'alphas_train'
        else:
            alpha_path = 'alphas_test'
        if alpha is not None:
            if not os.path.exists(alpha_path):
                os.mkdir(alpha_path)
            for i in range(10):
                plt.plot(alpha[i, :])
            plt.savefig(f"{alpha_path}/{col}.pdf")
            plt.clf()

    def plot_losses_over_examples(self, losses_over_examples, train, col):
        """
        plots the fixed point residuals over eval steps for each individual problem
        """
        if train:
            loe_folder = 'losses_over_examples_train'
        else:
            loe_folder = 'losses_over_examples_test'
        if not os.path.exists(loe_folder):
            os.mkdir(loe_folder)

        plt.plot(losses_over_examples)
        plt.yscale('log')
        plt.savefig(f"{loe_folder}/losses_{col}_plot.pdf", bbox_inches='tight')
        plt.clf()

    def plot_angles(self, angles, r, train, col):
        # SRG-type plots
        if train:
            polar_path = 'polar_train'
        else:
            polar_path = 'polar_test'
        if not os.path.exists(polar_path):
            os.mkdir(polar_path)
        if not os.path.exists(f"{polar_path}/{col}"):
            os.mkdir(f"{polar_path}/{col}")

        # plotting subsequent vectors in polar form
        for i in range(5):
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            radii = r[i, 1:] / r[i, :-1]
            theta = angles[i, :]
            ax.plot(theta, radii)
            ax.plot(theta[self.train_unrolls], radii[self.train_unrolls], 'r+')
            ax.grid(True)
            ax.set_rscale('symlog')
            ax.set_title("Magnitude", va='bottom')
            plt.legend()
            plt.savefig(f"{polar_path}/{col}/prob_{i}_subseq_mag.pdf")
            plt.clf()

        # save the angle data (or the cos(angle) data) for subseq.
        # new csv file for each
        subsequent_angles = angles
        angles_df = pd.DataFrame(subsequent_angles)
        angles_df.to_csv(f"{polar_path}/{col}/angle_data.csv")

        # also plot the angles for the first 5 problems
        for i in range(5):
            plt.plot(angles[i, :])
            plt.ylabel('angle')
            plt.xlabel('eval iters')
            plt.hlines(0, 0, angles[i, :].size, 'r')
            plt.savefig(f"{polar_path}/{col}/prob_{i}_angles.pdf")
            plt.clf()

    def plot_warm_starts(self, u_all, z_all, train, col):
        """
        plots the warm starts for the given method

        we give plots for
            x: primal variable
            y: dual variable
            z: base Douglas-Rachford iterate (dual of primal-dual variable)

        train is a boolean

        plots the first 5 problems and

        self.plot_iterates is a list
            e.g. [0, 10, 20]
            tells us to plot
                (z^0, z^10, z^20, z_opt) for each of the first 5 problems
                AND do a separate plot for
                (z^0 - z_opt, z^10 - z_opt, z^20 - z_opt) for each of the first 5 problems
        """
        if train:
            ws_path = 'warm-starts_train'
        else:
            ws_path = 'warm-starts_test'
        if not os.path.exists(ws_path):
            os.mkdir(ws_path)
        if not os.path.exists(f"{ws_path}/{col}"):
            os.mkdir(f"{ws_path}/{col}")
        # m, n = self.l2ws_model.m, self.l2ws_model.n
        for i in range(5):
            # if self.l2ws_model.hsde:
            #     x_hats, y_hats = u_all[i, :, :n], u_all[i, :, n:]
            # else:
            #     x_hats, y_hats = u_all[i, :, :n], u_all[i, :, n:]

            # # plot for x
            # for j in self.plot_iterates:
            #     plt.plot(u_all[i, j, :n], label=f"prediction_{j}")
            # if train:
            #     plt.plot(self.x_stars_train[i, :], label='optimal')
            # else:
            #     plt.plot(self.x_stars_test[i, :], label='optimal')
            # plt.legend()
            # plt.savefig(f"{ws_path}/{col}/prob_{i}_x_ws.pdf")
            # plt.clf()

            # for j in self.plot_iterates:
            #     plt.plot(u_all[i, j, :n] -
            #              self.x_stars_train[i, :], label=f"prediction_{j}")
            # plt.legend()
            # plt.title('diffs to optimal')
            # plt.savefig(f"{ws_path}/{col}/prob_{i}_diffs_x.pdf")
            # plt.clf()

            # # plot for y

            # for j in self.plot_iterates:
            #     plt.plot(u_all[i, j, n:n + m], label=f"prediction_{j}")
            # if train:
            #     plt.plot(self.y_stars_train[i, :], label='optimal')
            # else:
            #     plt.plot(self.y_stars_test[i, :], label='optimal')
            # plt.legend()
            # plt.savefig(f"{ws_path}/{col}/prob_{i}_y_ws.pdf")
            # plt.clf()

            # for j in self.plot_iterates:
            #     plt.plot(u_all[i, j, n:m + n] -
            #              self.y_stars_train[i, :], label=f"prediction_{j}")
            # plt.legend()
            # plt.title('diffs to optimal')
            # plt.savefig(f"{ws_path}/{col}/prob_{i}_diffs_y.pdf")
            # plt.clf()

            # plot for z
            for j in self.plot_iterates:
                plt.plot(z_all[i, j, :], label=f"prediction_{j}")
            if train:
                plt.plot(self.l2ws_model.z_stars_train[i, :], label='optimal')
            else:
                plt.plot(self.l2ws_model.z_stars_test[i, :], label='optimal')
            plt.legend()
            plt.savefig(f"{ws_path}/{col}/prob_{i}_z_ws.pdf")
            plt.clf()

            for j in self.plot_iterates:
                if isinstance(self.l2ws_model, OSQPmodel):
                    try:
                        plt.plot(z_all[i, j, :self.l2ws_model.m + self.l2ws_model.n] - self.l2ws_model.z_stars_train[i, :],
                                label=f"prediction_{j}")
                    except:
                        plt.plot(z_all[i, j, :self.l2ws_model.m + self.l2ws_model.n] - self.l2ws_model.z_stars_train[i, :self.l2ws_model.m + self.l2ws_model.n],
                                label=f"prediction_{j}")
                else:
                    plt.plot(z_all[i, j, :] - self.l2ws_model.z_stars_train[i, :],
                             label=f"prediction_{j}")
            plt.legend()
            plt.title('diffs to optimal')
            plt.savefig(f"{ws_path}/{col}/prob_{i}_diffs_z.pdf")
            plt.clf()
