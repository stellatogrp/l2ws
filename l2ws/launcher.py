from jax.config import config
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from l2ws.l2ws_model import L2WSmodel
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
import hydra
import time
import jax
from scipy.spatial import distance_matrix
from l2ws.algo_steps import create_projection_fn, get_psd_sizes
from l2ws.utils.generic_utils import sample_plot, setup_permutation
import scs
from scipy.sparse import csc_matrix
from l2ws.algo_steps import lin_sys_solve
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    "font.size": 16,
})
config.update("jax_enable_x64", True)


class Workspace:
    def __init__(self, cfg, static_flag, static_dict, example, get_M_q,
                 low_2_high_dim=None,
                 x_psd_indices=None,
                 y_psd_indices=None,
                 custom_visualize_fn=None):
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
        self.angle_anchors = cfg.angle_anchors
        self.supervised = cfg.supervised
        self.tx, self.ty = cfg.get('tx'), cfg.get('ty')
        self.dx, self.dy = cfg.get('dx'), cfg.get('dy')
        self.learn_XY = cfg.get('learn_XY')
        self.num_clusters = cfg.get('num_clusters')
        self.loss_method = cfg.loss_method
        self.plot_iterates = cfg.plot_iterates
        self.share_all = cfg.get('share_all')
        self.pretrain_alpha = cfg.get('pretrain_alpha')
        self.test_every_x_epochs = cfg.get('test_every_x_epochs')
        self.normalize_inputs = cfg.get('normalize_inputs')
        self.normalize_alpha = cfg.get('normalize_alpha')
        self.plateau_decay = cfg.plateau_decay
        self.epochs_jit = cfg.epochs_jit
        self.accs = cfg.get('accuracies')
        self.iterates_visualize = cfg.get('iterates_visualize')

        # custom visualization
        if custom_visualize_fn is None:
            self.has_custom_visualization = False
        else:
            self.has_custom_visualization = True
            self.custom_visualize_fn = custom_visualize_fn

        # from the run cfg retrieve the following via the data cfg
        self.nn_cfg = cfg.nn_cfg
        N_train, N_test = cfg.N_train, cfg.N_test
        N = N_train + N_test

        # load the data from problem to problem
        orig_cwd = hydra.utils.get_original_cwd()
        folder = f"{orig_cwd}/outputs/{example}/aggregate_outputs/{cfg.data.datetime}"
        filename = f"{folder}/data_setup_aggregate.npz"
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

        self.thetas_train = thetas[:N_train, :]
        self.thetas_test = thetas[N_train:N, :]

        w_stars_train = w_stars[:N_train, :]
        self.x_stars_test = x_stars[N_train:N, :]
        self.y_stars_test = y_stars[N_train:N, :]
        w_stars_test = w_stars[N_train:N, :]
        m = y_stars_train.shape[1]
        n = x_stars_train[0, :].size

        if static_flag:
            static_M = static_dict['M']

            static_algo_factor = static_dict['algo_factor']
            # cones_array = static_dict['cones_array']
            # cones = dict(z=int(cones_array[0]), l=int(cones_array[1]))
            cones = static_dict['cones_dict']

            # call get_q_mat
            q_mat = get_M_q(thetas)
            M_tensor_train, M_tensor_test = None, None
            matrix_invs_train, matrix_invs_test = None, None

            M_plus_I = static_M + jnp.eye(n + m)
            static_algo_factor = jsp.linalg.lu_factor(M_plus_I)
        else:
            # load the algo_factors -- check if factor or inverse
            M_tensor, q_mat = get_M_q(thetas)

            # load the matrix invs
            matrix_invs = jnp_load_obj['matrix_invs']

            static_M, static_algo_factor = None, None

            cones = static_dict['cones_dict']
            M_tensor_train = M_tensor[:N_train, :, :]
            M_tensor_test = M_tensor[N_train:N, :, :]
            matrix_invs_train = matrix_invs[:N_train, :, :]
            matrix_invs_test = matrix_invs[N_train:N, :, :]

        # save cones
        self.cones = static_dict['cones_dict']

        # alternate -- load it if available (but this is memory-intensive)
        q_mat_train = q_mat[:N_train, :]
        q_mat_test = q_mat[N_train:N, :]
        print('q_mat_train', q_mat_train[0, :])

        self.M = static_M

        self.train_unrolls = cfg.train_unrolls
        eval_unrolls = cfg.train_unrolls

        self.proj = create_projection_fn(cones, n)

        psd_sizes = get_psd_sizes(cones)

        # self.proj, psd_sizes = out[0], out[1]
        self.psd_size = psd_sizes[0]
        sdp_bool = self.psd_size > 0

        # normalize the inputs if the option is on
        if self.normalize_inputs:
            col_sums = thetas.mean(axis=0)
            inputs_normalized = (thetas - col_sums) / thetas.std(axis=0)
            inputs = jnp.array(inputs_normalized)
        else:
            inputs = thetas
        train_inputs = inputs[:N_train, :]
        test_inputs = inputs[N_train:N, :]

        num_plot = 5
        sample_plot(thetas, 'theta', num_plot)
        sample_plot(train_inputs, 'input', num_plot)
        sample_plot(x_stars, 'x_stars', num_plot)
        sample_plot(y_stars, 'y_stars', num_plot)
        sample_plot(w_stars, 'w_stars', num_plot)

        input_dict = {'nn_cfg': self.nn_cfg,
                      'proj': self.proj,
                      'train_inputs': train_inputs,
                      'test_inputs': test_inputs,
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
                      'y_stars_test': self.y_stars_test,
                      'x_stars_train': x_stars_train,
                      'x_stars_test': self.x_stars_test,
                      'static_flag': static_flag,
                      'static_algo_factor': static_algo_factor,
                      'matrix_invs_train': matrix_invs_train,
                      'matrix_invs_test': matrix_invs_test,
                      'angle_anchors': self.angle_anchors,
                      'supervised': self.supervised,
                      'psd': sdp_bool,
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
                      'loss_method': self.loss_method,
                      'share_all': self.share_all,
                      'pretrain_alpha': self.pretrain_alpha,
                      'normalize_alpha': self.normalize_alpha,
                      'plateau_decay': self.plateau_decay
                      }

        self.l2ws_model = L2WSmodel(input_dict)

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
        fixed_ws = col == 'nearest_neighbor'

        # do the actual evaluation (most important step in thie method)
        eval_out = self.evaluate_only(fixed_ws, num, train, col)

        # extract information from the evaluation
        loss_train, out_train, train_time = eval_out
        iter_losses_mean = out_train[2].mean(axis=0)
        angles = out_train[3]
        primal_residuals = out_train[4].mean(axis=0)
        dual_residuals = out_train[5].mean(axis=0)

        # plot losses over examples
        losses_over_examples = out_train[2].T
        self.plot_losses_over_examples(losses_over_examples, train, col)

        # update the eval csv files
        df_out = self.update_eval_csv(
            iter_losses_mean, primal_residuals, dual_residuals, train, col)
        iters_df, primal_residuals_df, dual_residuals_df = df_out

        # write accuracies dataframe to csv
        self.write_accuracies_csv(iter_losses_mean, train, col)

        # plot the evaluation iterations
        self.plot_eval_iters(iters_df, primal_residuals_df,
                             dual_residuals_df, plot_pretrain, train, col)

        # SRG-type plots
        r = out_train[2]
        self.plot_angles(angles, r, train, col)

        # plot the warm-start predictions
        u_all = out_train[0][3]
        z_all = out_train[0][0]
        self.plot_warm_starts(u_all, z_all, train, col)

        # plot the alpha coefficients
        alpha = out_train[0][2]
        self.plot_alphas(alpha, train, col)

        # custom visualize
        if self.has_custom_visualization:
            x_primals = u_all[:, :, :self.l2ws_model.n]
            self.custom_visualize(x_primals, train, col)

        # solve with scs
        # z0_mat = z_all[:, 0, :]
        # self.solve_scs(z0_mat, train, col)

        return out_train

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
        # solver = scs.SCS(scs_data, cones_dict)
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

        num = 20
        solve_times = np.zeros(num)

        if train:
            q_mat = self.l2ws_model.q_mat_train
        else:
            q_mat = self.l2ws_model.q_mat_test

        for i in range(num):
            # get the current q
            q = q_mat[i, :]

            # set b, c
            b, c = q_mat[i, n:], q_mat[i, :n]
            scs_data = dict(P=P, A=A, b=b, c=c)
            solver.update(b=np.array(b))
            solver.update(c=np.array(c))
            self.solver = solver

            # set the warm start
            x, y, s = self.get_xys_from_z(z0_mat[i, :], q)

            # solve
            sol = solver.solve(warm_start=True, x=np.array(x), y=np.array(y), s=np.array(s))

            # set the solve time in seconds
            solve_times[i] = sol['info']['solve_time'] / 1000

        # write the solve times to the csv file
        scs_df = pd.DataFrame(solve_times)
        scs_df.to_csv(f"{scs_path}/{col}/scs_solve_times.csv")

    def get_xys_from_z(self, z_init, q):
        n = self.l2ws_model.n
        u_tilde = lin_sys_solve(self.l2ws_model.static_algo_factor, z_init - q)
        u_temp = 2*u_tilde - z_init
        u = self.proj(u_temp)
        v = u + z_init - 2*u_tilde

        x, y = u[:n], u[n:]
        s = v[n:]
        return x, y, s

    def custom_visualize(self, x_primals, train, col):
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
            x_stars = self.l2ws_model.x_stars_train
            thetas = self.thetas_train
        else:
            x_stars = self.l2ws_model.x_stars_test
            thetas = self.thetas_test
        self.custom_visualize_fn(x_primals, x_stars, thetas, self.iterates_visualize, visual_path)

        # save x_primals to csv (TODO)
        # x_primals_df = pd.DataFrame(x_primals[:5, :])
        # x_primals_df.to_csv(f"{visualize_path}/{col}/x_primals.csv")

    def run(self):
        # setup logging and dataframes
        self._init_logging()
        self.setup_dataframes()

        # set pretrain_on boolean
        pretrain_on = self.pretrain_cfg.pretrain_iters > 0

        # no learning evaluation
        self.eval_iters_train_and_test('no_train', False)

        # fixed ws evaluation
        self.eval_iters_train_and_test('nearest_neighbor', False)

        # pretrain evaluation
        if pretrain_on:
            self.pretrain()

        # self.train_full()

        # eval test data to start
        self.test_eval_write()

        num_epochs_jit = int(self.l2ws_model.epochs / self.epochs_jit)

        # key_count updated to get random permutation for each epoch
        key_count = 0
        for epoch_batch in range(num_epochs_jit):
            epoch = int(epoch_batch * self.epochs_jit)
            if epoch % self.eval_every_x_epochs == 0:
                self.eval_iters_train_and_test(f"train_epoch_{epoch}", pretrain_on)
            if epoch > self.l2ws_model.dont_decay_until:
                self.l2ws_model.decay_upon_plateau()

            # setup the permutations
            permutation = setup_permutation(key_count, self.l2ws_model.N_train, self.epochs_jit)

            @jit
            def body_fn(batch, val):
                train_losses, params, state = val
                start_index = batch * self.l2ws_model.batch_size
                batch_indices = jax.lax.dynamic_slice(
                    permutation, (start_index,), (self.l2ws_model.batch_size,))
                train_loss, params, state = self.l2ws_model.train_batch(
                    batch_indices, params, state)
                train_losses = train_losses.at[batch].set(train_loss)
                val = train_losses, params, state
                return val

            epoch_batch_start_time = time.time()

            loop_size = int(self.l2ws_model.num_batches * self.epochs_jit)
            epoch_train_losses = jnp.zeros(loop_size)
            if epoch == 0:
                # unroll the first iterate so that This allows `init_val` and `body_fun`
                # below to have the same output type, which is a requirement of
                # jax.lax.while_loop and jax.lax.scan.
                batch_indices = jax.lax.dynamic_slice(
                    permutation, (0,), (self.l2ws_model.batch_size,))
                train_loss_first, params, state = self.l2ws_model.train_batch(
                    batch_indices, self.l2ws_model.params, self.l2ws_model.state)

                epoch_train_losses = epoch_train_losses.at[0].set(train_loss_first)
                start_index = 1
            else:
                start_index = 0

            # loop the last (self.l2ws_model.num_batches - 1) iterates
            init_val = epoch_train_losses, params, state
            val = jax.lax.fori_loop(start_index, loop_size, body_fn, init_val)

            epoch_batch_end_time = time.time()
            time_diff = epoch_batch_end_time - epoch_batch_start_time
            time_train_per_epoch = time_diff / self.epochs_jit
            epoch_train_losses, params, state = val

            # reset the global (params, state)
            self.l2ws_model.epoch += self.epochs_jit
            self.l2ws_model.params = params
            self.l2ws_model.state = state

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

    def write_accuracies_csv(self, losses, train, col):
        # def update_acc(df_acc, accs, col, losses):
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

        # save both iterations and fraction reduction in single table
        # df_acc_both = pd.DataFrame()
        # # df_acc_both['accuracies'] = df_acc['no_learn']
        # df_acc_both['no_learn_iters'] = np.array(self.accs)

        # for col in df_percent.columns:
        #     if col != 'accuracies' and col != 'no_learn':
        #         df_acc_both[col + '_iters'] = df_acc[col]
        #         df_acc_both[col + '_red'] = df_percent[col]
        # df_acc_both.to_csv(f"{accs_path}/{col}/accuracies_reduction_both.csv")

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

    def evaluate_only(self, fixed_ws, num, train, col):
        tag = 'train' if train else 'test'
        z_stars = self.l2ws_model.w_stars_train[:num,
                                                :] if train else self.l2ws_model.w_stars_test[:num,
                                                                                              :]
        q_mat = self.l2ws_model.q_mat_train[:num,
                                            :] if train else self.l2ws_model.q_mat_test[:num, :]

        factors = None if self.l2ws_model.static_flag else self.l2ws_model.matrix_invs_train[:num,
                                                                                             :]
        M_tensor = None if self.l2ws_model.static_flag else self.l2ws_model.M_tensor_train[:num, :]

        inputs = self.get_inputs_for_eval(fixed_ws, num, train, col)
        eval_out = self.l2ws_model.evaluate(self.eval_unrolls, inputs, factors,
                                            M_tensor, q_mat, z_stars, fixed_ws, tag=tag)
        return eval_out

    def get_inputs_for_eval(self, fixed_ws, num, train, col):
        if fixed_ws:
            inputs = self.get_nearest_neighbors(train, num)
        else:
            # elif col == 'no_train': (possible case to consider)
            # random init with neural network ()
            # _, predict_size = self.l2ws_model.w_stars_test.shape
            # random_start = np.random.normal(size=(num, predict_size))
            # inputs = jnp.array(random_start)
            # fixed_ws = True
            if train:
                inputs = self.l2ws_model.train_inputs[:num, :]
            else:
                inputs = self.l2ws_model.test_inputs[:num, :]
        return inputs

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
        return self.l2ws_model.w_stars_train[indices, :]

    def setup_dataframes(self):
        self.iters_df_train = pd.DataFrame(
            columns=['iterations', 'no_train'])
        self.iters_df_train['iterations'] = np.arange(1, self.eval_unrolls+1)

        self.iters_df_test = pd.DataFrame(
            columns=['iterations', 'no_train'])
        self.iters_df_test['iterations'] = np.arange(1, self.eval_unrolls+1)

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

    def update_eval_csv(self, iter_losses_mean, primal_residuals, dual_residuals, train, col):
        """
        update the eval csv files
        """
        if train:
            self.iters_df_train[col] = iter_losses_mean
            self.iters_df_train.to_csv('iters_compared_train.csv')
            self.primal_residuals_df_train[col] = primal_residuals
            self.primal_residuals_df_train.to_csv('primal_residuals_train.csv')
            self.dual_residuals_df_train[col] = dual_residuals
            self.dual_residuals_df_train.to_csv('dual_residuals_train.csv')

            iters_df = self.iters_df_train
            primal_residuals_df = self.primal_residuals_df_train
            dual_residuals_df = self.dual_residuals_df_train
        else:
            self.iters_df_test[col] = iter_losses_mean
            self.iters_df_test.to_csv('iters_compared_test.csv')
            self.primal_residuals_df_test[col] = primal_residuals
            self.primal_residuals_df_test.to_csv('primal_residuals_test.csv')
            self.dual_residuals_df_test[col] = dual_residuals
            self.dual_residuals_df_test.to_csv('dual_residuals_test.csv')

            iters_df = self.iters_df_test
            primal_residuals_df = self.primal_residuals_df_test
            dual_residuals_df = self.dual_residuals_df_test
        return iters_df, primal_residuals_df, dual_residuals_df

    def plot_eval_iters(self, iters_df, primal_residuals_df, dual_residuals_df, plot_pretrain,
                        train, col):
        # plot of the fixed point residuals
        plt.plot(iters_df['no_train'], 'k-', label='no learning')
        if col != 'no_train':
            plt.plot(iters_df['nearest_neighbor'], 'm-', label='nearest neighbor')
        if plot_pretrain:
            plt.plot(iters_df['pretrain'], 'r-', label='pretraining')
        if col != 'no_train' and col != 'pretrain' and col != 'fixed_ws':
            plt.plot(iters_df[col], label=f"train k={self.train_unrolls}")
        plt.yscale('log')
        plt.xlabel('evaluation iterations')
        plt.ylabel('test fixed point residuals')
        plt.legend()
        if train:
            plt.title('train problems')
            plt.savefig('eval_iters_train.pdf', bbox_inches='tight')
        else:
            plt.title('test problems')
            plt.savefig('eval_iters_test.pdf', bbox_inches='tight')
        plt.clf()

        # plot of the primal and dual residuals
        plt.plot(primal_residuals_df['no_train'],
                 'k+', label='no learning primal')
        plt.plot(dual_residuals_df['no_train'],
                 'ko', label='no learning dual')

        if plot_pretrain:
            plt.plot(
                primal_residuals_df['pretrain'], 'r+', label='pretraining primal')
            plt.plot(dual_residuals_df['pretrain'],
                     'ro', label='pretraining dual')
        if col != 'no_train' and col != 'pretrain' and col != 'fixed_ws':
            plt.plot(
                primal_residuals_df[col], label=f"train k={self.train_unrolls} primal")
            plt.plot(
                dual_residuals_df[col], label=f"train k={self.train_unrolls} dual")
        plt.yscale('log')
        plt.xlabel('evaluation iterations')
        plt.ylabel('test primal-dual residuals')
        plt.legend()
        plt.savefig('primal_dual_residuals.pdf', bbox_inches='tight')
        plt.clf()

    def plot_alphas(self, alpha, train, col):
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

        num_angles = len(self.angle_anchors)
        for i in range(5):
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            for j in range(num_angles):
                angle = self.angle_anchors[j]
                curr_r = r[i, angle:-1]
                theta = np.zeros(curr_r.size)
                theta[1:] = angles[i, j, angle+1:]
                ax.plot(theta, curr_r, label=f"anchor={angle}")
                ax.plot(theta[self.train_unrolls-angle], curr_r[self.train_unrolls-angle], 'r+')
            ax.grid(True)
            ax.set_rscale('symlog')
            ax.set_title("Magnitude", va='bottom')
            plt.legend()
            plt.savefig(f"{polar_path}/{col}/prob_{i}_mag.pdf")
            plt.clf()

        for i in range(5):
            fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
            for j in range(num_angles):
                angle = self.angle_anchors[j]
                curr_r = r[i, angle:-1]
                theta = np.zeros(curr_r.size)
                theta[1:] = angles[i, j, angle+1:]
                num_iters = np.max([100, self.train_unrolls + 5])
                r2 = num_iters - np.arange(num_iters)
                ax2.plot(theta[:num_iters], r2, label=f"anchor={angle}")
                ax2.plot(theta[self.train_unrolls-angle], r2[self.train_unrolls-angle], 'r+')
            ax2.grid(True)
            ax2.set_title("Iterations", va='bottom')
            plt.legend()
            plt.savefig(f"{polar_path}/{col}/prob_{i}_iters.pdf")
            plt.clf()

        '''
        plotting subsequent vectors in polar form
        '''
        num_angles = len(self.angle_anchors)
        for i in range(5):
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            curr_r = r[i, :-1]
            theta = np.zeros(curr_r.size)

            theta[1:] = angles[i, -1, 1:]
            ax.plot(theta, curr_r, label=f"anchor={angle}")
            ax.plot(theta[self.train_unrolls-angle], curr_r[self.train_unrolls-angle], 'r+')
            ax.grid(True)
            ax.set_rscale('symlog')
            ax.set_title("Magnitude", va='bottom')
            plt.legend()
            plt.savefig(f"{polar_path}/{col}/prob_{i}_subseq_mag.pdf")
            plt.clf()

        for i in range(5):
            fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
            for j in range(num_angles):
                curr_r = r[i, :-1]
                theta = np.zeros(curr_r.size)
                theta[1:] = angles[i, -1, 1:]
                num_iters = np.max([100, self.train_unrolls + 5])
                r2 = num_iters - np.arange(num_iters)
                ax2.plot(theta[:num_iters], r2, label=f"anchor={angle}")
                ax2.plot(theta[self.train_unrolls], r2[self.train_unrolls], 'r+')
            ax2.grid(True)
            ax2.set_title("Iterations", va='bottom')
            plt.legend()
            plt.savefig(f"{polar_path}/{col}/prob_{i}_subseq_iters.pdf")
            plt.clf()

        '''
        save the angle data (or the cos(angle) data) for subseq.
        - new csv file for each
        - put
        '''
        subsequent_angles = angles[:, -1, 1:]
        angles_df = pd.DataFrame(subsequent_angles)
        angles_df.to_csv(f"{polar_path}/{col}/angle_data.csv")

        # also plot the angles for the first 5 problems
        for i in range(5):
            plt.plot(angles[i, -1, 2:])
            plt.ylabel('angle')
            plt.xlabel('eval iters')
            plt.hlines(0, 0, angles[i, -1, 2:].size, 'r')
            plt.savefig(f"{polar_path}/{col}/prob_{i}_angles.pdf")
            plt.clf()

    def plot_warm_starts(self, u_all, z_all, train, col):
        if train:
            ws_path = 'warm-starts_train'
        else:
            ws_path = 'warm-starts_test'
        if not os.path.exists(ws_path):
            os.mkdir(ws_path)
        if not os.path.exists(f"{ws_path}/{col}"):
            os.mkdir(f"{ws_path}/{col}")
        for i in range(5):
            # plot for x
            for j in self.plot_iterates:
                plt.plot(u_all[i, j, :self.l2ws_model.n], label=f"prediction_{j}")
            if train:
                plt.plot(self.x_stars_train[i, :], label='optimal')
            else:
                plt.plot(self.x_stars_test[i, :], label='optimal')
            plt.legend()
            plt.savefig(f"{ws_path}/{col}/prob_{i}_x_ws.pdf")
            plt.clf()

            for j in self.plot_iterates:
                plt.plot(u_all[i, j, :self.l2ws_model.n] -
                         self.x_stars_train[i, :], label=f"prediction_{j}")
            plt.legend()
            plt.title('diffs to optimal')
            plt.savefig(f"{ws_path}/{col}/prob_{i}_diffs_x.pdf")
            plt.clf()

            # plot for y
            for j in self.plot_iterates:
                plt.plot(u_all[i, j, self.l2ws_model.n:], label=f"prediction_{j}")
            if train:
                plt.plot(self.y_stars_train[i, :], label='optimal')
            else:
                plt.plot(self.y_stars_test[i, :], label='optimal')
            plt.legend()
            plt.savefig(f"{ws_path}/{col}/prob_{i}_y_ws.pdf")
            plt.clf()

            for j in self.plot_iterates:
                plt.plot(u_all[i, j, self.l2ws_model.n:] -
                         self.y_stars_train[i, :], label=f"prediction_{j}")
            plt.legend()
            plt.title('diffs to optimal')
            plt.savefig(f"{ws_path}/{col}/prob_{i}_diffs_y.pdf")
            plt.clf()

            # plot for z
            for j in self.plot_iterates:
                plt.plot(z_all[i, j, :], label=f"prediction_{j}")
            if train:
                plt.plot(self.l2ws_model.w_stars_train[i, :], label='optimal')
            else:
                plt.plot(self.l2ws_model.w_stars_test[i, :], label='optimal')
            plt.legend()
            plt.savefig(f"{ws_path}/{col}/prob_{i}_z_ws.pdf")
            plt.clf()

            for j in self.plot_iterates:
                plt.plot(z_all[i, j, :] - self.l2ws_model.w_stars_train[i, :],
                         label=f"prediction_{j}")
            plt.legend()
            plt.title('diffs to optimal')
            plt.savefig(f"{ws_path}/{col}/prob_{i}_diffs_z.pdf")
            plt.clf()
