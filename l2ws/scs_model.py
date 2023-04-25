from l2ws.l2ws_model import L2WSmodel
import time
import jax.numpy as jnp
from l2ws.algo_steps import k_steps_eval, k_steps_train, lin_sys_solve
from functools import partial
from jax import vmap, jit


class SCSmodel(L2WSmodel):
    def __init__(self, input_dict):
        super(SCSmodel, self).__init__(input_dict)

    def setup_optimal_solutions(self, dict):
        if dict.get('x_stars_train', None) is not None:
            self.y_stars_train, self.y_stars_test = dict['y_stars_train'], dict['y_stars_test']
            self.x_stars_train, self.x_stars_test = dict['x_stars_train'], dict['x_stars_test']
            self.z_stars_train = jnp.array(dict['z_stars_train'])
            self.z_stars_test = jnp.array(dict['z_stars_test'])
            self.u_stars_train = jnp.hstack([self.x_stars_train, self.y_stars_train])
            self.u_stars_test = jnp.hstack([self.x_stars_test, self.y_stars_test])
        else:
            self.z_stars_train, self.z_stars_test = None, None

    def train_batch(self, batch_indices, params, state):
        batch_inputs = self.train_inputs[batch_indices, :]
        batch_q_data = self.q_mat_train[batch_indices, :]
        batch_z_stars = self.z_stars_train[batch_indices, :] if self.supervised else None
        results = self.optimizer.update(params=params,
                                        state=state,
                                        inputs=batch_inputs,
                                        q=batch_q_data,
                                        iters=self.train_unrolls,
                                        z_stars=batch_z_stars)
        params, state = results
        return state.value, params, state

    def short_test_eval(self):
        z_stars_test = self.z_stars_test if self.supervised else None
        if self.static_flag:
            test_loss, test_out, time_per_prob = self.static_eval(self.train_unrolls,
                                                                  self.test_inputs,
                                                                  self.q_mat_test,
                                                                  z_stars_test)
        else:
            eval_out = self.dynamic_eval(self.train_unrolls,
                                         self.test_inputs,
                                         self.matrix_invs_test,
                                         self.M_tensor_test,
                                         self.q_mat_test)
            test_loss, test_out, time_per_prob = eval_out
        self.te_losses.append(test_loss)

        time_per_iter = time_per_prob / self.train_unrolls
        return test_loss, time_per_iter

    def evaluate(self, k, inputs, matrix_inv, M, q, z_stars, fixed_ws, tag='test'):
        if self.static_flag:
            return self.static_eval(k, inputs, q, z_stars, tag=tag, fixed_ws=fixed_ws)
        else:
            return self.dynamic_eval(k, inputs, matrix_inv, M, q, tag=tag, fixed_ws=fixed_ws)
        

    def static_eval(self, k, inputs, q, z_stars, tag='test', fixed_ws=False):
        if fixed_ws:
            curr_loss_fn = self.loss_fn_fixed_ws
        else:
            curr_loss_fn = self.loss_fn_eval
        num_probs, _ = inputs.shape

        test_time0 = time.time()

        loss, out = curr_loss_fn(
            self.params, inputs, q, k, z_stars)
        time_per_prob = (time.time() - test_time0)/num_probs
        print('eval time per prob', time_per_prob)
        print(f"[Epoch {self.epoch}] [k {k}] {tag} loss: {loss:.6f}")
        return loss, out, time_per_prob

    def dynamic_eval(self, k, inputs, matrix_inv, M, q, tag='test', fixed_ws=False):
        if fixed_ws:
            curr_loss_fn = self.loss_fn_fixed_ws
        else:
            curr_loss_fn = self.loss_fn_eval
        num_probs, _ = inputs.shape
        test_time0 = time.time()
        loss, out = curr_loss_fn(
            self.params, inputs, matrix_inv, M, q, k)
        time_per_prob = (time.time() - test_time0)/num_probs
        print('eval time per prob', time_per_prob)
        print(f"[Epoch {self.epoch}] [k {k}] {tag} loss: {loss:.6f}")

        return loss, out, time_per_prob
    
    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = self.supervised and diff_required

        proj, n, m = self.proj, self.n, self.m
        M_static, factor_static = self.static_M, self.static_algo_factor
        share_all = self.share_all
        Z_shared = self.Z_shared if share_all else None
        normalize_alpha = self.normalize_alpha if share_all else None
        loss_method, static_flag = self.loss_method, self.static_flag
        hsde, jit = self.hsde, self.jit
        zero_cone_size, rho_x = self.zero_cone_size, self.rho_x
        scale, alpha_relax = self.scale, self.alpha_relax

        def predict(params, input, q, iters, z_star, factor, M):
            P, A = M[:n, :n], -M[n:, :n]
            b, c = q[n:], q[:n]

            z0, alpha = self.predict_warm_start(params, input, bypass_nn, hsde,
                                           share_all, Z_shared, normalize_alpha)
            if hsde:
                q_r = lin_sys_solve(factor, q)
            else:
                q_r = q

            if diff_required:
                z_final, iter_losses = k_steps_train(
                    iters, z0, q_r, factor, supervised, z_star, proj, jit, hsde, m, n,
                    zero_cone_size, rho_x, scale, alpha_relax)
            else:
                k_eval_out = k_steps_eval(
                    iters, z0, q_r, factor, proj, P, A, c, b, jit, hsde,
                    zero_cone_size, rho_x, scale, alpha_relax)
                z_final, iter_losses = k_eval_out[0], k_eval_out[1]
                primal_residuals, dual_residuals = k_eval_out[2], k_eval_out[3]
                all_z_plus_1, all_u, all_v = k_eval_out[4], k_eval_out[5], k_eval_out[6]

                # compute angle(z^{k+1} - z^k, z^k - z^{k-1})
                diffs = jnp.diff(all_z_plus_1, axis=0)
                angles = self.batch_angle(diffs[:-1], diffs[1:])

            loss = self.final_loss(loss_method, z_final, iter_losses, supervised, z0, z_star)

            if diff_required:
                return loss
            else:
                out = all_z_plus_1, z_final, alpha, all_u, all_v
                return loss, iter_losses, angles, primal_residuals, dual_residuals, out
        loss_fn = self.predict_2_loss(predict, static_flag, diff_required, factor_static, M_static)
        return loss_fn
    
    def create_static_loss_fn(self, predict, diff_required, factor_static, M_static):
        out_axes = self.get_out_axes_shape(diff_required)
        predict_final = partial(predict,
                                factor=factor_static,
                                M=M_static
                                )
        batch_predict = vmap(predict_final, in_axes=(
            None, 0, 0, None, 0), out_axes=out_axes)

        @partial(jit, static_argnums=(3,))
        def loss_fn(params, inputs, q, iters, z_stars):
            if diff_required:
                losses = batch_predict(params, inputs, q, iters, z_stars)
                return losses.mean()
            else:
                predict_out = batch_predict(
                    params, inputs, q, iters, z_stars)
                losses, iter_losses, angles, primal_residuals, dual_residuals, out = predict_out
                loss_out = out, losses, iter_losses, angles, primal_residuals, dual_residuals
                return losses.mean(), loss_out
        return loss_fn


    def create_dynamic_loss_fn(self, predict, diff_required):
        out_axes = self.get_out_axes_shape(diff_required)
        batch_predict = vmap(predict, in_axes=(
            None, 0, 0, None, 0, 0), out_axes=out_axes)

        @partial(jit, static_argnums=(5,))
        def loss_fn(params, inputs, factor, M, q, iters):
            if diff_required:
                losses = batch_predict(params, inputs, q, iters, factor, M)
                return losses.mean()
            else:
                predict_out = batch_predict(
                    params, inputs, q, iters, factor, M)
                losses, iter_losses, angles, primal_residuals, dual_residuals, out = predict_out
                loss_out = out, losses, iter_losses, angles, primal_residuals, dual_residuals
                return losses.mean(), loss_out
        return loss_fn
    
    def predict_2_loss(self, predict, static_flag, diff_required, factor_static, M_static):
        """
        given the predict fn this returns the loss fn

        basically breaks the prediction fn into multiple cases
            - diff_required used for training, but not evaluation
            - static_flag is True if the matrices (P, A) are the same for each problem
                factor_static and M_static are shared for all problems and passed in

        for the evaluation, we store a lot more information
        for the training, we store nothing - just return the loss
            this could be changed - need to turn the boolean has_aux=True
                in self.optimizer = OptaxSolver(opt=optax.adam(
                    self.lr), fun=self.loss_fn_train, has_aux=False)

        in all forward passes, the number of iterations is static
        """
        if static_flag:
            loss_fn = self.create_static_loss_fn(predict, diff_required, factor_static, M_static)
        else:
            loss_fn = self.create_dynamic_loss_fn(predict, diff_required)
        return loss_fn


    def get_out_axes_shape(self, diff_required):
        if diff_required:
            # out_axes for (loss)
            out_axes = (0)
        else:
            # out_axes for
            #   (loss, iter_losses, angles, primal_residuals, dual_residuals, out)
            #   out = (all_z_, z_next, alpha, all_u, all_v)
            out_axes = (0, 0, 0, 0, 0, (0, 0, 0, 0, 0))
        return out_axes
