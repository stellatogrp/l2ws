from jax.config import config
from ast import Return
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from l2ws.l2ws_model import L2WSmodel
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap
import hydra
import pdb
import time
import jax
from jax import random, lax
from l2ws.scs_problem import SCSinstance, scs_jax
from scipy.spatial import distance_matrix
from utils.generic_utils import vec_symm, unvec_symm
from functools import partial
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    "font.size": 16,
})
config.update("jax_enable_x64", True)


def soc_proj_single(input):
    y, s = input[1:], input[0]
    pi_y, pi_s = soc_projection(y, s)
    return jnp.append(pi_s, pi_y)


def sdp_proj_single(x, dim):
    X = unvec_symm(x, dim)
    evals, evecs = jnp.linalg.eigh(X)
    evals_plus = jnp.clip(evals, 0, jnp.inf)
    num_proj = evals_plus - evals > 0
    print('evals diff', num_proj.sum())
    X_proj = evecs @ jnp.diag(evals_plus) @ evecs.T
    x_proj = vec_symm(X_proj)
    return x_proj


def soc_projection(y, s):
    y_norm = jnp.linalg.norm(y)

    def case1_soc_proj(y, s):
        # case 1: y_norm >= |s|
        val = (s + y_norm) / (2 * y_norm)
        t = val * y_norm
        x = val * y
        return x, t

    def case2_soc_proj(y, s):
        # case 2: y_norm <= |s|
        # case 2a: s > 0
        # case 2b: s < 0
        def case2a(y, s):
            return y, s

        def case2b(y, s):
            return (0.0*jnp.zeros(2), 0.0)
        return lax.cond(s >= 0, case2a, case2b, y, s)
    return lax.cond(y_norm >= jnp.abs(s), case1_soc_proj, case2_soc_proj, y, s)


class Workspace:
    def __init__(self, cfg, static_flag, static_dict, example, get_M_q,
                 low_2_high_dim=None,
                 x_psd_indices=None,
                 y_psd_indices=None):
        '''
        cfg is the run_cfg
        static_dict holds the data that doesn't change from problem to problem
        '''
        self.cfg = cfg
        self.itr = 0
        self.eval_unrolls = cfg.eval_unrolls
        self.eval_every_x_epochs = cfg.eval_every_x_epochs
        self.save_every_x_epochs = cfg.save_every_x_epochs
        self.num_samples = cfg.num_samples
        self.pretrain_cfg = cfg.pretrain
        self.prediction_variable = cfg.prediction_variable
        self.angle_anchors = cfg.angle_anchors
        self.supervised = cfg.supervised
        self.tx = cfg.tx
        self.ty = cfg.ty
        self.dx = cfg.dx
        self.dy = cfg.dy
        self.learn_XY = cfg.learn_XY
        self.num_clusters = cfg.num_clusters
        self.loss_method = cfg.loss_method
        self.plot_iterates = cfg.plot_iterates

        '''
        from the run cfg retrieve the following via the data cfg
        '''
        self.nn_cfg = cfg.nn_cfg
        N_train, N_test = cfg.N_train, cfg.N_test
        N = N_train + N_test

        # load the data from problem to problem
        orig_cwd = hydra.utils.get_original_cwd()
        filename = f"{orig_cwd}/outputs/{example}/aggregate_outputs/{cfg.data.datetime}/data_setup_aggregate.npz"
        jnp_load_obj = jnp.load(filename)

        '''
        previously was saving + loading the common data
        but that was too memory-intensive
        so just passing it in now
        '''
        thetas = jnp_load_obj['thetas']
        x_stars = jnp_load_obj['x_stars']
        y_stars = jnp_load_obj['y_stars']
        w_stars = jnp_load_obj['w_stars']
        x_stars_train = x_stars[:N_train, :]
        y_stars_train = y_stars[:N_train, :]

        self.x_stars_train = x_stars[:N_train, :]
        self.y_stars_train = y_stars[:N_train, :]
        w_stars_train = w_stars[:N_train, :]
        x_stars_test = x_stars[N_train:N, :]
        y_stars_test = y_stars[N_train:N, :]
        w_stars_test = w_stars[N_train:N, :]
        m = y_stars_train.shape[1]
        n = x_stars_train[0, :].size
        for i in range(4):
            plt.plot(thetas[i, :])
        plt.savefig('sample_thetas.pdf')
        plt.clf()

        if static_flag:
            static_M = static_dict['M']

            static_algo_factor = static_dict['algo_factor']
            # cones_array = static_dict['cones_array']
            # cones = dict(z=int(cones_array[0]), l=int(cones_array[1]))
            cones = static_dict['cones_dict']

            # call get_q_mat
            q_mat = get_M_q(thetas)
            dynamic_algo_factors, M_tensor = None, None
            M_tensor_train, M_tensor_test = None, None
            matrix_invs_train, matrix_invs_test = None, None

            M_plus_I = static_M + jnp.eye(n + m)
            static_algo_factor = jsp.linalg.lu_factor(M_plus_I)
        else:
            # load the algo_factors -- check if factor or inverse
            M_tensor, q_mat = get_M_q(thetas)

            # load the matrix invs
            matrix_invs = jnp_load_obj['matrix_invs']

            # calculate matrix invs
            # def inv(M_in):
            #     return jnp.linalg.inv(M_in + jnp.eye(m+n))
            # batch_inv  = vmap(inv, in_axes=(0), out_axes=(0))
            # t0 = time.time()
            # matrix_invs = batch_inv(M_tensor)
            # t1 = time.time()
            # print('inv time', t1 - t0)
            static_M, static_algo_factor = None, None
            cones_array = static_dict['cones_array']
            # cones = dict(z=cones_array[0], l=cones_array[1])
            cones = static_dict['cones_dict']
            M_tensor_train = M_tensor[:N_train, :, :]
            M_tensor_test = M_tensor[N_train:N, :, :]
            matrix_invs_train = matrix_invs[:N_train, :, :]
            matrix_invs_test = matrix_invs[N_train:N, :, :]

        # alternate -- load it if available (but this is memory-intensive)
        q_mat_train = q_mat[:N_train, :]
        q_mat_test = q_mat[N_train:N, :]
        print('q_mat_train', q_mat_train[0, n:])

        self.M = static_M

        zero_cone, nonneg_cone = cones['z'], cones['l']

        soc = 'q' in cones.keys() and len(cones['q']) > 0
        sdp_ = 's' in cones.keys() and len(cones['s']) > 0

        self.train_unrolls = cfg.train_unrolls
        eval_unrolls = cfg.train_unrolls

        zero_cone_int = int(zero_cone)
        nonneg_cone_int = int(nonneg_cone)
        if soc:
            num_soc = len(cones['q'])
            soc_total = sum(cones['q'])
            soc_cones_array = np.array(cones['q'])
            soc_size = soc_cones_array[0]
            soc_proj_single_batch = vmap(soc_proj_single, in_axes=(0), out_axes=(0))
        else:
            soc_total = 0
        if sdp_:
            num_sdp = len(cones['s'])
            sdp_total = sum(cones['s'])
            sdp_cones_array = np.array(cones['s'])
            sdp_size = int(sdp_cones_array[0] * (sdp_cones_array[0]+1) / 2)
            sdp_proj_single_dim = partial(sdp_proj_single, dim=sdp_cones_array[0])
            sdp_proj_single_batch = vmap(sdp_proj_single_dim, in_axes=(0), out_axes=(0))

        @jit
        def proj(input):
            nonneg = jnp.clip(input[n+zero_cone_int:n+zero_cone_int+nonneg_cone_int], a_min=0)
            projection = jnp.concatenate([input[:n+zero_cone_int], nonneg])
            if soc:
                socp = jnp.zeros(soc_total)
                soc_input = input[n+zero_cone_int+nonneg_cone_int:n +
                                  zero_cone_int+nonneg_cone_int+soc_total]
                soc_input_reshaped = jnp.reshape(soc_input, (num_soc, soc_size))
                soc_out_reshaped = soc_proj_single_batch(soc_input_reshaped)
                socp = jnp.ravel(soc_out_reshaped)
                projection = jnp.concatenate([projection, socp])
            if sdp_:
                sdp = jnp.zeros(sdp_total)
                sdp_input = input[n + zero_cone_int+nonneg_cone_int+soc_total:]
                sdp_input_reshaped = jnp.reshape(sdp_input, (num_sdp, sdp_size))
                sdp_out_reshaped = sdp_proj_single_batch(sdp_input_reshaped)
                sdp = jnp.ravel(sdp_out_reshaped)
                projection = jnp.concatenate([projection, sdp])
            return projection

        self.proj = proj


        # @jit
        # def lin_sys_solve(rhs):
        #     return jsp.linalg.lu_solve(algo_factor, rhs)

        def lin_sys_solve(factor_, rhs):
            if static_flag:
                return jsp.linalg.lu_solve(factor_, rhs)
            else:
                return factor_ @ rhs
        self.lin_sys_solve = lin_sys_solve

        # normalize the inputs
        col_sums = thetas.mean(axis=0)
        inputs_normalized = (thetas - col_sums) / thetas.std(axis=0)
        inputs = jnp.array(inputs_normalized)
        train_inputs = inputs[:N_train, :]
        test_inputs = inputs[N_train:N, :]
        # train_inputs = thetas[:N_train, :]
        # test_inputs = thetas[N_train:N, :]

        num_plot = np.min([N_train, 4])
        for i in range(num_plot):
            plt.plot(x_stars_train[i, :])
        plt.savefig('sample_x_stars.pdf')
        plt.clf()

        for i in range(num_plot):
            plt.plot(y_stars_train[i, :])
        plt.savefig('sample_y_stars.pdf')
        plt.clf()

        # check
        # P_jax = static_M[:n,:n]
        # A_jax = -static_M[n:,:n]
        # c_jax, b_jax = q_mat_train[0,:n], q_mat_train[0,n:]
        # data = dict(P=P_jax, A=A_jax, b=b_jax, c=c_jax, cones=cones)
        # data['x'] = x_stars[0, :]
        # data['y'] = y_stars[0, :]
        # x_jax, y_jax, s_jax = scs_jax(data, iters=1000)
        # pdb.set_trace()
        # end check

        cones = static_dict['cones_dict']
        self.psd_size = cones['s'][0]  # TOFIX in general

        input_dict = {'nn_cfg': self.nn_cfg,
                      'proj': proj,
                      'train_inputs': train_inputs,
                      'test_inputs': test_inputs,
                      'lin_sys_solve': lin_sys_solve,
                      'train_unrolls': self.train_unrolls,
                      'eval_unrolls': eval_unrolls,
                      'w_stars_train': w_stars_train,
                      'w_stars_test': w_stars_test,
                      'q_mat_train': q_mat_train,
                      'q_mat_test': q_mat_test,
                      'M_tensor_train': M_tensor_train,
                      'M_tensor_test': M_tensor_test,
                      'm': m,
                      'n': n,
                      'static_M': static_M,
                      'y_stars_train': y_stars_train,
                      'y_stars_test': y_stars_test,
                      'x_stars_train': x_stars_train,
                      'x_stars_test': x_stars_test,
                      'prediction_variable': self.prediction_variable,
                      'static_flag': static_flag,
                      'static_algo_factor': static_algo_factor,
                      'matrix_invs_train': matrix_invs_train,
                      'matrix_invs_test': matrix_invs_test,
                      #   'dynamic_algo_factors': matrix_invs,
                      'angle_anchors': self.angle_anchors,
                      'supervised': self.supervised,
                      'psd': sdp_,
                      'tx': self.tx,
                      'ty': self.ty,
                      'dx': self.dx,
                      'dy': self.dy,
                      'psd_size': self.psd_size,
                      'low_2_high_dim': low_2_high_dim,
                      'learn_XY': self.learn_XY,
                      'num_clusters': self.num_clusters,
                      'x_psd_indices': x_psd_indices,
                      'y_psd_indices': y_psd_indices,
                      'loss_method': self.loss_method
                      }

        self.l2ws_model = L2WSmodel(input_dict)

    def _init_logging(self):
        self.logf = open('log.csv', 'a')
        fieldnames = ['iter', 'train_loss', 'val_loss', 'test_loss']
        self.writer = csv.DictWriter(self.logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            self.writer.writeheader()

    def evaluate_iters(self, num, col, train=False, plot=True, plot_pretrain=False):
        fixed_ws = col == 'fixed_ws'
        if train:
            if fixed_ws:
                '''
                find closest train point
                neural network input
                '''

                # compute distances to train inputs
                distances = distance_matrix(
                    np.array(self.l2ws_model.train_inputs[:num, :]), np.array(self.l2ws_model.train_inputs))
                print('distances', distances)
                indices = np.argmin(distances, axis=1)
                print('indices', indices)
                best_val = np.min(distances, axis=1)
                print('best val', best_val)
                plt.plot(indices)
                plt.savefig(f"indices_plot.pdf", bbox_inches='tight')
                plt.clf()

                inputs = self.l2ws_model.w_stars_train[indices, :]

            elif col == 'no_train':
                # random init with neural network
                _, predict_size = self.l2ws_model.w_stars_test.shape
                random_start = 0*np.random.normal(size=(num, predict_size))
                inputs = jnp.array(random_start)
                fixed_ws = True

                # inputs = self.l2ws_model.train_inputs[:num, :]
                # fixed_ws = False
            else:
                inputs = self.l2ws_model.train_inputs[:num, :]
            if self.l2ws_model.static_flag:
                eval_out = self.l2ws_model.evaluate(self.eval_unrolls,
                                                    # self.l2ws_model.train_inputs[:num, :],
                                                    inputs,
                                                    self.l2ws_model.q_mat_train[:num, :],
                                                    self.l2ws_model.w_stars_train[:num, :],
                                                    tag='train',
                                                    fixed_ws=fixed_ws)
            else:
                eval_out = self.l2ws_model.dynamic_eval(self.eval_unrolls,
                                                        # self.l2ws_model.train_inputs[:num, :],
                                                        inputs,
                                                        self.l2ws_model.matrix_invs_train[:num, :],
                                                        self.l2ws_model.M_tensor_train[:num, :],
                                                        self.l2ws_model.q_mat_train[:num, :],
                                                        tag='train',
                                                        fixed_ws=fixed_ws)
        else:
            if fixed_ws:
                '''
                find closest train point
                neural network input
                '''

                # compute distances to train inputs
                distances = distance_matrix(
                    np.array(self.l2ws_model.test_inputs[:num, :]), np.array(self.l2ws_model.train_inputs))
                print('distances', distances)
                indices = np.argmin(distances, axis=1)
                print('indices', indices)
                best_val = np.min(distances, axis=1)
                print('best val', best_val)
                plt.plot(indices)
                plt.savefig(f"indices_plot.pdf", bbox_inches='tight')
                plt.clf()

                inputs = self.l2ws_model.w_stars_train[indices, :]

            elif col == 'no_train':
                # random init with neural network
                # _, predict_size = self.l2ws_model.w_stars_test.shape
                # random_start = .05*np.random.normal(size=(num, predict_size))
                # inputs = jnp.array(random_start)
                # fixed_ws = True

                #
                inputs = self.l2ws_model.test_inputs[:num, :]
                fixed_ws = False
            else:
                inputs = self.l2ws_model.test_inputs[:num, :]
            if self.l2ws_model.static_flag:
                eval_out = self.l2ws_model.evaluate(self.eval_unrolls,
                                                    inputs,
                                                    self.l2ws_model.q_mat_test[:num, :],
                                                    tag='test',
                                                    fixed_ws=fixed_ws)
            else:
                eval_out = self.l2ws_model.dynamic_eval(self.eval_unrolls,
                                                        inputs,
                                                        self.l2ws_model.matrix_invs_test[:num, :, :],
                                                        self.l2ws_model.M_tensor_test[:num, :, :],
                                                        self.l2ws_model.q_mat_test[:num, :],
                                                        tag='test',
                                                        fixed_ws=fixed_ws)

        loss_train, out_train, train_time = eval_out
        iter_losses_mean = out_train[2].mean(axis=0)
        if not os.path.exists('losses_over_examples'):
            os.mkdir('losses_over_examples')
        plt.plot(out_train[2].T)
        plt.yscale('log')
        plt.savefig(f"losses_over_examples/losses_{col}_plot.pdf", bbox_inches='tight')
        plt.clf()

        angles = out_train[3]
        primal_residuals = out_train[4].mean(axis=0)
        dual_residuals = out_train[5].mean(axis=0)
        print('after iterations z', out_train[0][1][0, :])
        print('truth z', self.l2ws_model.w_stars_test[0, :])
        print('after iterations z', out_train[0][1][1, :])
        print('truth z', self.l2ws_model.w_stars_test[1, :])
        # plt.plot(out_train[0][1][0,:], label='after iters')
        # plt.plot(self.l2ws_model.w_stars_test[0, :], label='truth')
        plt.plot(self.l2ws_model.w_stars_test[0, :] - out_train[0][1][0, :], label='truth')
        plt.savefig('debug.pdf', bbox_inches='tight')
        plt.clf()

        # if not train:
        self.iters_df[col] = iter_losses_mean
        self.iters_df.to_csv('iters_compared.csv')
        self.primal_residuals_df[col] = primal_residuals
        self.primal_residuals_df.to_csv('primal_residuals.csv')
        self.dual_residuals_df[col] = dual_residuals
        self.dual_residuals_df.to_csv('dual_residuals.csv')

        '''
        now save the plots so we can monitor
        -- no_learning colummn
        -- pretrain column (if enabled)
        -- last column of training
        '''

        # plot of the fixed point residuals
        plt.plot(self.iters_df['no_train'], 'k-', label='no learning')
        if col != 'no_train':
            plt.plot(self.iters_df['fixed_ws'], 'm-', label='naive warm start')
        if plot_pretrain:
            plt.plot(self.iters_df['pretrain'], 'r-', label='pretraining')
        if col != 'no_train' and col != 'pretrain' and col != 'fixed_ws':
            plt.plot(self.iters_df[col], label=f"train k={self.train_unrolls}")
        plt.yscale('log')
        plt.xlabel('evaluation iterations')
        plt.ylabel('test fixed point residuals')
        plt.legend()
        plt.savefig('eval_iters.pdf', bbox_inches='tight')
        plt.clf()

        # plot of the primal and dual residuals
        plt.plot(self.primal_residuals_df['no_train'],
                 'k+', label='no learning primal')
        plt.plot(self.dual_residuals_df['no_train'],
                 'ko', label='no learning dual')

        if plot_pretrain:
            plt.plot(
                self.primal_residuals_df['pretrain'], 'r+', label='pretraining primal')
            plt.plot(self.dual_residuals_df['pretrain'],
                     'ro', label='pretraining dual')
        if col != 'no_train' and col != 'pretrain' and col != 'fixed_ws':
            plt.plot(
                self.primal_residuals_df[col], label=f"train k={self.train_unrolls} primal")
            plt.plot(
                self.dual_residuals_df[col], label=f"train k={self.train_unrolls} dual")
        plt.yscale('log')
        plt.xlabel('evaluation iterations')
        plt.ylabel('test primal-dual residuals')
        plt.legend()
        plt.savefig('primal_dual_residuals.pdf', bbox_inches='tight')
        plt.clf()

        # SRG-type plots
        # one for each problem
        if not os.path.exists('polar'):
            os.mkdir('polar')
        if not os.path.exists(f"polar/{col}"):
            os.mkdir(f"polar/{col}")

        num_angles = len(self.angle_anchors)
        for i in range(5):
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            # fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
            for j in range(num_angles):
                angle = self.angle_anchors[j]
                r = out_train[2][i, angle:-1]
                theta = np.zeros(r.size)
                theta[1:] = angles[i, j, angle+1:]
                ax.plot(theta, r, label=f"anchor={angle}")
                ax.plot(theta[self.train_unrolls-angle], r[self.train_unrolls-angle], 'r+')
            ax.grid(True)
            ax.set_rscale('symlog')
            ax.set_title("Magnitude", va='bottom')
            plt.legend()
            plt.savefig(f"polar/{col}/prob_{i}_mag.pdf")
            plt.clf()

        for i in range(5):
            fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
            for j in range(num_angles):
                angle = self.angle_anchors[j]
                r = out_train[2][i, angle:-1]
                theta = np.zeros(r.size)
                theta[1:] = angles[i, j, angle+1:]
                num_iters = np.max([100, self.train_unrolls + 5])
                r2 = num_iters - np.arange(num_iters)
                ax2.plot(theta[:num_iters], r2, label=f"anchor={angle}")
                ax2.plot(theta[self.train_unrolls-angle], r2[self.train_unrolls-angle], 'r+')
            ax2.grid(True)
            # ax2.set_rscale('symlog')
            ax2.set_title("Iterations", va='bottom')
            plt.legend()
            plt.savefig(f"polar/{col}/prob_{i}_iters.pdf")
            plt.clf()


        '''
        plotting subsequent vectors in polar form
        '''
        num_angles = len(self.angle_anchors)
        for i in range(5):
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            r = out_train[2][i, 0:-1]
            theta = np.zeros(r.size)
            theta[1:] = angles[i, -1, 1:]
            ax.plot(theta, r, label=f"anchor={angle}")
            ax.plot(theta[self.train_unrolls-angle], r[self.train_unrolls-angle], 'r+')
            ax.grid(True)
            ax.set_rscale('symlog')
            ax.set_title("Magnitude", va='bottom')
            plt.legend()
            plt.savefig(f"polar/{col}/prob_{i}_subseq_mag.pdf")
            plt.clf()

        for i in range(5):
            fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
            for j in range(num_angles):
                r = out_train[2][i, 0:-1]
                theta = np.zeros(r.size)
                theta[1:] = angles[i, -1, 1:]
                num_iters = np.max([100, self.train_unrolls + 5])
                r2 = num_iters - np.arange(num_iters)
                ax2.plot(theta[:num_iters], r2, label=f"anchor={angle}")
                ax2.plot(theta[self.train_unrolls], r2[self.train_unrolls], 'r+')
            ax2.grid(True)
            ax2.set_title("Iterations", va='bottom')
            plt.legend()
            plt.savefig(f"polar/{col}/prob_{i}_subseq_iters.pdf")
            plt.clf()

        '''
        save the angle data (or the cos(angle) data) for subseq.
        - new csv file for each
        - put
        '''

        # if not os.path.exists(f"polar/angle_data/{col}"):
        #     os.mkdir(f"polar/angle_data/{col}")
        subsequent_angles = angles[:, -1, 1:]
        angles_df = pd.DataFrame(subsequent_angles)
        angles_df.to_csv(f"polar/{col}/angle_data.csv")

        # also plot the angles for the first 5 problems
        for i in range(5):
            plt.plot(angles[i, -1, 2:])
            plt.ylabel('angle')
            plt.xlabel('eval iters')
            plt.hlines(0, 0, angles[i, -1, 2:].size, 'r')
            plt.savefig(f"polar/{col}/prob_{i}_angles.pdf")
            plt.clf()

        '''
        plot the warm-start predictions
        '''
        u_ws = out_train[0][0]
        u_all = out_train[0][3]

        if not os.path.exists('warm-starts'):
            os.mkdir('warm-starts')
        if not os.path.exists(f"warm-starts/{col}"):
            os.mkdir(f"warm-starts/{col}")
        for i in range(5):
            '''
            plot for x
            '''
            if train:
                plt.plot(self.x_stars_train[i, :], label='optimal')
            else:
                plt.plot(self.x_stars_test[i, :], label='optimal')
            for j in self.plot_iterates:
                plt.plot(u_all[i, j, :self.l2ws_model.n], label=f"prediction_{j}")
            plt.legend()
            plt.savefig(f"warm-starts/{col}/prob_{i}_x_ws.pdf")
            plt.clf()

            for j in self.plot_iterates:
                plt.plot(u_all[i, j, :self.l2ws_model.n] - self.x_stars_train[i, :], label=f"prediction_{j}")
            plt.legend()
            plt.title('diffs to optimal')
            plt.savefig(f"warm-starts/{col}/prob_{i}_diffs_x.pdf")
            plt.clf()


            '''
            plot for y
            '''
            if train:
                plt.plot(self.y_stars_train[i, :], label='optimal')
            else:
                plt.plot(self.y_stars_test[i, :], label='optimal')

            for j in self.plot_iterates:
                plt.plot(u_all[i, j, self.l2ws_model.n:], label=f"prediction_{j}")
            plt.legend()
            plt.savefig(f"warm-starts/{col}/prob_{i}_y_ws.pdf")
            plt.clf()

            for j in self.plot_iterates:
                plt.plot(u_all[i, j, self.l2ws_model.n:] - self.y_stars_train[i, :], label=f"prediction_{j}")
            plt.legend()
            plt.title('diffs to optimal')
            plt.savefig(f"warm-starts/{col}/prob_{i}_diffs_y.pdf")
            plt.clf()

        return out_train

    def run(self):
        self._init_logging()

        self.iters_df = pd.DataFrame(
            columns=['iterations', 'no_train', 'final'])
        self.iters_df['iterations'] = np.arange(1, self.eval_unrolls+1)
        self.primal_residuals_df = pd.DataFrame(
            columns=['iterations'])
        self.primal_residuals_df['iterations'] = np.arange(
            1, self.eval_unrolls+1)
        self.dual_residuals_df = pd.DataFrame(
            columns=['iterations'])
        self.dual_residuals_df['iterations'] = np.arange(
            1, self.eval_unrolls+1)

        '''
        no learning evaluation
        '''
        pretrain_on = self.pretrain_cfg.pretrain_iters > 0
        out_train_start = self.evaluate_iters(
            self.num_samples, 'no_train', train=True, plot_pretrain=False)

        '''
        fixed ws evaluation
        '''
        out_train_fixed_ws = self.evaluate_iters(
            self.num_samples, 'fixed_ws', train=True, plot_pretrain=False)

        if pretrain_on:
            print("Pretraining...")
            self.df_pretrain = pd.DataFrame(
                columns=['pretrain_loss', 'pretrain_test_loss'])
            train_pretrain_losses, test_pretrain_losses = self.l2ws_model.pretrain(self.pretrain_cfg.pretrain_iters,
                                                                                   stepsize=self.pretrain_cfg.pretrain_stepsize,
                                                                                   df_pretrain=self.df_pretrain,
                                                                                   batches=self.pretrain_cfg.pretrain_batches)
            out_train_fixed_ws = self.evaluate_iters(
                self.num_samples, 'pretrain', train=True, plot_pretrain=pretrain_on)
            plt.plot(train_pretrain_losses, label='train')
            plt.plot(test_pretrain_losses, label='test')
            plt.yscale('log')
            plt.xlabel('pretrain iterations')
            plt.ylabel('pretrain loss')
            plt.legend()
            plt.savefig('pretrain_losses.pdf')
            plt.clf()

        self.logf = open('train_results.csv', 'a')
        fieldnames = ['iter', 'train_loss', 'moving_avg_train', 'test_loss', 'time_per_iter']
        self.writer = csv.DictWriter(self.logf, fieldnames=fieldnames)
        if os.stat('train_results.csv').st_size == 0:
            self.writer.writeheader()

        out_trains = []

        '''
        NEW WAY - train_batch - better for saving
        '''
        curr_iter = 0
        for epoch in range(self.l2ws_model.epochs):

            key = random.PRNGKey(epoch)
            permutation = jax.random.permutation(key, self.l2ws_model.N_train)
            for batch in range(self.l2ws_model.num_batches):
                start_index = batch*self.l2ws_model.batch_size
                end_index = (batch+1)*self.l2ws_model.batch_size
                batch_indices = permutation[start_index:end_index]

                if epoch % self.nn_cfg.decay_every == 0 and epoch > 0:
                    decay_lr_flag = True
                else:
                    decay_lr_flag = False
                self.l2ws_model.train_batch(
                    batch_indices, decay_lr_flag=decay_lr_flag,
                    writer=self.writer, logf=self.logf)

                # self.writer.writerow({
                #     'iter': self.state.iter_num,
                #     'train_loss': self.state.value,
                #     'moving_avg_train': moving_avg_train,
                #     'test_loss': test_loss,
                #     'time_per_iter': time_per_iter
                # })

                curr_iter += 1
            if epoch % self.eval_every_x_epochs == 0:
                out_train = self.evaluate_iters(
                    self.num_samples, f"train_iter_{curr_iter}", train=True, plot_pretrain=pretrain_on)
                # out_trains.append(out_train)
            self.l2ws_model.epoch += 1

            # plot the train / test loss so far
            if epoch % self.save_every_x_epochs == 0:
                batch_losses = np.array(self.l2ws_model.tr_losses_batch)
                te_losses = np.array(self.l2ws_model.te_losses)
                num_data_points = batch_losses.size
                epoch_axis = np.arange(num_data_points) / \
                    self.l2ws_model.num_batches
                plt.plot(epoch_axis, batch_losses, label='train')
                plt.plot(epoch_axis, te_losses, label='test')
                plt.yscale('log')
                plt.xlabel('epochs')
                plt.ylabel('fixed point residual average')
                plt.legend()
                plt.savefig('losses_over_training.pdf', bbox_inches='tight')
                plt.clf()

                plt.plot(epoch_axis, batch_losses, label='train')
                # plt.plot(epoch_axis, te_losses, label='test')
                plt.yscale('log')
                plt.xlabel('epochs')
                plt.ylabel('fixed point residual average')
                plt.legend()
                plt.savefig('train_losses_over_training.pdf', bbox_inches='tight')
                plt.clf()
