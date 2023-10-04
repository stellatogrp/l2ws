import jax.numpy as jnp
from jax import lax


def train_jitted_epochs(model, permutation, num_batches, epochs_jit, epoch=0):
    """
    train self.epochs_jit at a time
    special case: the first time we call train_batch (i.e. epoch = 0)
    """
    def train_over_epochs_body_simple_fn(batch, val):
        """
        to be used as the body_fn in lax.fori_loop
        need to call partial for the specific permutation
        """
        train_losses, params, state, permutation = val
        start_index = batch * model.batch_size
        batch_indices = lax.dynamic_slice(
            permutation, (start_index,), (model.batch_size,))
        train_loss, params, state = model.train_batch(
            batch_indices, params, state)
        train_losses = train_losses.at[batch].set(train_loss)
        val = train_losses, params, state, permutation
        return val

    # epoch_batch_start_time = time.time()
    loop_size = int(num_batches * epochs_jit)
    epoch_train_losses = jnp.zeros(loop_size)
    if epoch == 0:
        # unroll the first iterate so that This allows `init_val` and `body_fun`
        #   below to have the same output type, which is a requirement of
        #   lax.while_loop and lax.scan.
        batch_indices = lax.dynamic_slice(
            permutation, (0,), (model.batch_size,))
        
        train_loss_first, params, state = model.train_batch(
            batch_indices, model.params, model.state)

        epoch_train_losses = epoch_train_losses.at[0].set(train_loss_first)
        start_index = 1
        train_over_epochs_body_simple_fn_jitted = train_over_epochs_body_simple_fn
    else:
        start_index = 0
        params, state = model.params, model.state

    init_val = epoch_train_losses, params, state, permutation

    val = lax.fori_loop(start_index, loop_size, train_over_epochs_body_simple_fn_jitted, init_val)
    epoch_train_losses, params, state, permutation = val
    return params, state, epoch_train_losses