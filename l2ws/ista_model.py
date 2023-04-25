from l2ws.l2ws_model import L2WSmodel
import time
import jax.numpy as jnp
from l2ws.algo_steps import k_steps_eval_ista, k_steps_train_ista, k_steps_eval_fista, k_steps_train_fista
from functools import partial
from jax import vmap, jit


class ISTAmodel(L2WSmodel):
    def __init__(self, input_dict):
        self.fista = input_dict['algorithm'] == 'fista'
        super(ISTAmodel, self).__init__(input_dict)

    def setup_optimal_solutions(self, dict):
        if dict.get('z_stars_train', None) is not None:
            self.z_stars_train = jnp.array(dict['z_stars_train'])
            self.z_stars_test = jnp.array(dict['z_stars_test'])
        else:
            self.z_stars_train, self.z_stars_test = None, None
        

    def train_batch(self, batch_indices, params, state):
        batch_inputs = self.train_inputs[batch_indices, :]
        batch_b_data = self.b_mat_train[batch_indices, :]
        batch_z_stars = self.z_stars_train[batch_indices, :] if self.supervised else None
        results = self.optimizer.update(params=params,
                                        state=state,
                                        inputs=batch_inputs,
                                        b=batch_b_data,
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
    
    def static_eval(self, k, inputs, b, z_stars, tag='test', fixed_ws=False):
        if fixed_ws:
            curr_loss_fn = self.loss_fn_fixed_ws
        else:
            curr_loss_fn = self.loss_fn_eval
        num_probs, _ = inputs.shape

        test_time0 = time.time()

        loss, out = curr_loss_fn(self.params, inputs, b, k, z_stars)
        time_per_prob = (time.time() - test_time0)/num_probs
        # print('eval time per prob', time_per_prob)
        # print(f"[Epoch {self.epoch}] [k {k}] {tag} loss: {loss:.6f}")
        return loss, out, time_per_prob

    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = self.supervised #and diff_required

        jit = self.jit
        share_all = self.share_all
        loss_method = self.loss_method

        ista_step = self.ista_step
        lambd = self.lambd
        A = self.A

        def predict(params, input, b, iters, z_star):
            z0, alpha = self.predict_warm_start(params, input, bypass_nn)


            if diff_required:
                if self.fista:
                    z_final, iter_losses = k_steps_train_fista(iters, z0, b, lambd, A,
                                                          ista_step, supervised, z_star, jit)
                else:
                    z_final, iter_losses = k_steps_train_ista(iters, z0, b, lambd, A,
                                                          ista_step, supervised, z_star, jit)
            else:
                if self.fista:
                    z_final, iter_losses, z_all_plus_1 = k_steps_eval_fista(iters, z0, b, lambd, A,
                                                                       ista_step, supervised, z_star, jit)
                else:
                    z_final, iter_losses, z_all_plus_1 = k_steps_eval_ista(iters, z0, b, lambd, A,
                                                                       ista_step, supervised, z_star, jit)

                # compute angle(z^{k+1} - z^k, z^k - z^{k-1})
                diffs = jnp.diff(z_all_plus_1, axis=0)
                angles = self.batch_angle(diffs[:-1], diffs[1:])

            loss = self.final_loss(loss_method, z_final, iter_losses, supervised, z0, z_star)

            if diff_required:
                return loss
            else:
                # out = z_all_plus_1, z_final
                return loss, iter_losses, angles, z_all_plus_1
        # loss_fn = predict_2_loss(predict, static_flag, diff_required, factor_static, M_static)
        loss_fn = self.predict_2_loss_ista(predict, diff_required)
        return loss_fn

    def predict_2_loss_ista(self, predict, diff_required):
        out_axes = self.get_out_axes_shape_ista(diff_required)
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

    def get_out_axes_shape_ista(self, diff_required):
        if diff_required:
            # out_axes for (loss)
            out_axes = (0)
        else:
            # out_axes for
            #   (loss, iter_losses, angles, primal_residuals, dual_residuals, out)
            #   out = (all_z_, z_next, alpha, all_u, all_v)
            out_axes = (0, 0, 0, 0)
        return out_axes
