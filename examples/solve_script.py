import numpy as np
import logging
import yaml
from jax import vmap
import pandas as pd
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp
import os
import scs
import jax.scipy as jsp
from l2ws.scs_problem import SCSinstance


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
    }
)
log = logging.getLogger(__name__)


def setup_script(q_mat, theta_mat, solver, data, cones_dict, output_filename):
    N = q_mat.shape[0]
    m, n = data['A'].shape

    solve_times = np.zeros(N)
    x_stars = jnp.zeros((N, n))
    y_stars = jnp.zeros((N, m))
    s_stars = jnp.zeros((N, m))
    q_mat = jnp.zeros((N, m + n))
    # scs_instances = []

    P_sparse, A_sparse = data['P_sparse'], data['A_sparse']
    for i in range(N):
        log.info(f"solving problem number {i}")

        # update
        b = np.array(q_mat[i, n:])
        c = np.array(q_mat[i, :n])

        # manual canon
        manual_canon_dict = {
            "P": P_sparse,
            "A": A_sparse,
            "b": b,
            "c": c,
            "cones": cones_dict,
        }
        scs_instance = SCSinstance(manual_canon_dict, solver, manual_canon=True)

        # scs_instances.append(scs_instance)
        x_stars = x_stars.at[i, :].set(scs_instance.x_star)
        y_stars = y_stars.at[i, :].set(scs_instance.y_star)
        s_stars = s_stars.at[i, :].set(scs_instance.s_star)
        q_mat = q_mat.at[i, :].set(scs_instance.q)
        solve_times[i] = scs_instance.solve_time

        if i % 1000 == 0:
            log.info(f"saving final data... after solving problem number {i}")
            jnp.savez(
                output_filename,
                thetas=theta_mat,
                x_stars=x_stars,
                y_stars=y_stars,
                s_stars=s_stars,
                q_mat=q_mat
            )
    # save the data
    log.info("final saving final data...")
    t0 = time.time()
    jnp.savez(
        output_filename,
        thetas=theta_mat,
        x_stars=x_stars,
        y_stars=y_stars,
        s_stars=s_stars,
        q_mat=q_mat
    )
