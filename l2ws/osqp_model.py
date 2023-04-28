from l2ws.l2ws_model import L2WSmodel
import time
import jax.numpy as jnp
from l2ws.algo_steps import k_steps_eval_osqp, k_steps_train_osqp
from functools import partial
from jax import vmap, jit


class OSQPmodel(L2WSmodel):
    def __init__(self, input_dict):
        self.fista = input_dict['algorithm'] == 'fista'
        super(OSQPmodel, self).__init__(input_dict)

    def initialize_algo(self, input_dict):
        self.A = input_dict['A']
        if self.static_flag:
            self.static_M = input_dict['static_M']
            self.static_algo_factor = input_dict['static_algo_factor']
        else:
            pass
            
        self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']
        self.rho = input_dict.get('rho', 1)
        self.sigma = input_dict.get('sigma', 1)

        self.m, self.n = self.A.shape
        self.output_size = self.n + self.m
        

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
    
    def evaluate(self, k, inputs, b, z_stars, fixed_ws, tag='test'):
        return self.static_eval(k, inputs, b, z_stars, tag=tag, fixed_ws=fixed_ws)

    def short_test_eval(self):
        z_stars_test = self.z_stars_test if self.supervised else None
        
        test_loss, test_out, time_per_prob = self.static_eval(self.train_unrolls,
                                                              self.test_inputs,
                                                              self.b_mat_test,
                                                              z_stars_test)
        self.te_losses.append(test_loss)

        time_per_iter = time_per_prob / self.train_unrolls
        return test_loss, time_per_iter
    
    def static_eval(self, k, inputs, q, z_stars, tag='test', fixed_ws=False):
        if fixed_ws:
            curr_loss_fn = self.loss_fn_fixed_ws
        else:
            curr_loss_fn = self.loss_fn_eval
        num_probs, _ = inputs.shape

        test_time0 = time.time()

        loss, out = curr_loss_fn(self.params, inputs, q, k, z_stars)
        time_per_prob = (time.time() - test_time0)/num_probs
        # print('eval time per prob', time_per_prob)
        # print(f"[Epoch {self.epoch}] [k {k}] {tag} loss: {loss:.6f}")
        return loss, out, time_per_prob

    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = self.supervised #and diff_required

        jit = self.jit
        share_all = self.share_all
        loss_method = self.loss_method

        sigma = self.sigma
        rho = self.rho
        A = self.A
        factor = self.factor

        def predict(params, input, q, iters, z_star):
            xy0, alpha = self.predict_warm_start(params, input, bypass_nn)

            if diff_required:
                z_final, iter_losses = k_steps_train_osqp(iters, xy0, factor, A, q, rho, sigma, 
                                                          supervised, z_star, jit)
            else:
                z_final, iter_losses, z_all_plus_1 = k_steps_eval_osqp(iters, xy0, factor, A, q, rho, sigma, 
                                                                       supervised, z_star, jit)

                # compute angle(z^{k+1} - z^k, z^k - z^{k-1})
                diffs = jnp.diff(z_all_plus_1, axis=0)
                angles = self.batch_angle(diffs[:-1], diffs[1:])

            loss = self.final_loss(loss_method, z_final, iter_losses, supervised, xy0, z_star)

            if diff_required:
                return loss
            else:
                # out = z_all_plus_1, z_final
                return loss, iter_losses, angles, z_all_plus_1
        # loss_fn = predict_2_loss(predict, static_flag, diff_required, factor_static, M_static)
        loss_fn = self.predict_2_loss_osqp(predict, diff_required)
        return loss_fn

    def predict_2_loss_osqp(self, predict, diff_required):
        out_axes = self.get_out_axes_shape_osqp(diff_required)
        batch_predict = vmap(predict,
                             in_axes=(None, 0, 0, None, 0),
                             out_axes=out_axes)

        @partial(jit, static_argnums=(3,))
        def loss_fn(params, inputs, b, iters, z_stars):
            if diff_required:
                losses = batch_predict(params, inputs, b, iters, z_stars)
                return losses.mean()
            else:
                predict_out = batch_predict(
                    params, inputs, b, iters, z_stars)
                losses, iter_losses, angles, z_all = predict_out
                loss_out = losses, iter_losses, angles, z_all
                # return losses.mean(), losses, iter_losses, angles, z_all
                return losses.mean(), loss_out
        return loss_fn

    # def get_out_axes_shape_osqp(self, diff_required):
    #     if diff_required:
    #         # out_axes for (loss)
    #         out_axes = (0)
    #     else:
    #         # out_axes for
    #         #   (loss, iter_losses, angles, primal_residuals, dual_residuals, out)
    #         #   out = (all_z_, z_next, alpha, all_u, all_v)
    #         out_axes = (0, 0, 0, 0)
    #     return out_axes
