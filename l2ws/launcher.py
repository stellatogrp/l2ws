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
from jax import random
from l2ws.scs_problem import SCSinstance, scs_jax
from scipy.spatial import distance_matrix
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    "font.size": 16,
})


class Workspace:
    def __init__(self, cfg, static_flag, static_dict, example, get_M_q):
        '''
        cfg is the run_cfg
        static_dict holds the data that doesn't change from problem to problem
        '''
        self.cfg = cfg
        self.itr = 0
        self.eval_unrolls = cfg.eval_unrolls
        self.eval_every_x_epochs = cfg.eval_every_x_epochs
        self.num_samples = cfg.num_samples
        self.pretrain_cfg = cfg.pretrain
        self.prediction_variable = cfg.prediction_variable

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
            cones_array = static_dict['cones_array']
            cones = dict(z=int(cones_array[0]), l=int(cones_array[1]))

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
            cones = dict(z=cones_array[0], l=cones_array[1])
            M_tensor_train = M_tensor[:N_train, :, :]
            M_tensor_test = M_tensor[N_train:N, :, :]
            matrix_invs_train = matrix_invs[:N_train, :, :]
            matrix_invs_test = matrix_invs[N_train:N, :, :]

        # alternate -- load it if available (but this is memory-intensive)
        q_mat_train = q_mat[:N_train, :]
        q_mat_test = q_mat[N_train:N, :]
        print('q_mat_train', q_mat_train[0,n:])

        self.M = static_M

        zero_cone, pos_cone = cones['z'], cones['l']

        self.train_unrolls = cfg.train_unrolls
        eval_unrolls = cfg.train_unrolls

        zero_cone_int = int(zero_cone)

        @jit
        def proj(input):
            proj = jnp.clip(input[n+zero_cone_int:], a_min=0)
            return jnp.concatenate([input[:n+zero_cone_int], proj])
        self.proj = proj

        # pdb.set_trace()
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
        for i in range(4):
            plt.plot(x_stars_train[i, :])
        plt.savefig('sample_problems.pdf')
        plt.clf()

        ########### check
        # P_jax = static_M[:n,:n]
        # A_jax = -static_M[n:,:n]
        # c_jax, b_jax = q_mat_train[0,:n], q_mat_train[0,n:]
        # data = dict(P=P_jax, A=A_jax, b=b_jax, c=c_jax, cones=cones)
        # data['x'] = x_stars[0, :]
        # data['y'] = y_stars[0, :]
        # x_jax, y_jax, s_jax = scs_jax(data, iters=1000)
        # pdb.set_trace()
        ########### end check

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
            if self.l2ws_model.static_flag:
                eval_out = self.l2ws_model.evaluate(self.eval_unrolls,
                                                   self.l2ws_model.train_inputs[:num, :],
                                                   self.l2ws_model.q_mat_train[:num, :],
                                                   tag='train')
            else:
                eval_out = self.l2ws_model.dynamic_eval(self.eval_unrolls,
                                                   self.l2ws_model.train_inputs[:num, :],
                                                   self.l2ws_model.matrix_invs_train[:num, :],
                                                   self.l2ws_model.M_tensor_train[:num, :],
                                                   self.l2ws_model.q_mat_train[:num, :],
                                                   tag='train')
        else:
            if fixed_ws:
                '''
                find closest train point
                neural network input
                '''

                # compute distances to train inputs
                distances = distance_matrix(np.array(self.l2ws_model.test_inputs[:num,:]), np.array(self.l2ws_model.train_inputs))
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

        primal_residuals = out_train[3].mean(axis=0)
        dual_residuals = out_train[4].mean(axis=0)
        print('after iterations z', out_train[0][1][0,:])
        print('truth z', self.l2ws_model.w_stars_test[0, :])
        print('after iterations z', out_train[0][1][1,:])
        print('truth z', self.l2ws_model.w_stars_test[1, :])
        # plt.plot(out_train[0][1][0,:], label='after iters')
        # plt.plot(self.l2ws_model.w_stars_test[0, :], label='truth')
        plt.plot(self.l2ws_model.w_stars_test[0, :] - out_train[0][1][0,:], label='truth')
        plt.savefig('debug.pdf', bbox_inches='tight')
        plt.clf()

        if not train:
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
            plt.plot(self.iters_df['pretrain'], 'r+', label='pretraining')
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
        out_train_start = self.evaluate_iters(
            self.num_samples, 'no_train', train=False, plot_pretrain=False)

        '''
        fixed ws evaluation
        '''
        out_train_fixed_ws = self.evaluate_iters(
            self.num_samples, 'fixed_ws', train=False, plot_pretrain=False)
        

        # print("Pretraining...")
        # self.df_pretrain = pd.DataFrame(
        #     columns=['pretrain_loss', 'pretrain_test_loss'])
        # train_pretrain_losses, test_pretrain_losses = self.l2ws_model.pretrain(self.pretrain_cfg.pretrain_iters,
        #                                                                       stepsize=self.pretrain_cfg.pretrain_stepsize,
        #                                                                       df_pretrain=self.df_pretrain)
        # plt.plot(train_pretrain_losses, label='train')
        # plt.plot(test_pretrain_losses, label='test')
        # plt.yscale('log')
        # plt.xlabel('pretrain iterations')
        # plt.ylabel('pretrain loss')
        # plt.legend()
        # plt.savefig('pretrain_losses.pdf')
        # plt.clf()

        self.logf = open('train_results.csv', 'a')
        fieldnames = ['iter', 'train_loss', 'moving_avg_train', 'test_loss']
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

                curr_iter += 1
            if epoch % self.eval_every_x_epochs == 0:
                out_train = self.evaluate_iters(
                    self.num_samples, f"train_iter_{curr_iter}", train=False)
                # out_trains.append(out_train)
            self.l2ws_model.epoch += 1

            # plot the train / test loss so far
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
