import functools
import sys
import hydra
import os
import jax.numpy as jnp
import pdb
from jax import jit, vmap
import examples.markowitz as markowitz
import examples.osc_mass as osc_mass
import examples.vehicle as vehicle
import yaml
import numpy as np
import time
from utils.data_utils import recover_last_datetime
# from l2ws.scs_problem import SCSinstance, scs_jax

@hydra.main(config_path='configs/markowitz', config_name='markowitz_agg.yaml')
def markowitz_main(cfg):
    '''
    given data and time of all the previous experiments
    creates new folder with hydra
    combines all the npz files into 1
    '''
    example = 'markowitz'
    orig_cwd = hydra.utils.get_original_cwd()
    datetimes = cfg.datetimes
    if len(datetimes) == 0:
        # get the most recent datetime and update datetimes
        last_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        datetimes = [last_datetime]

        cfg = {'datetimes': datetimes}
        
        # save the datetime to we can recover
        with open('agg_datetimes.yaml', 'w') as file:
            yaml.dump(cfg, file)

    datetime0 = datetimes[0]
    static_flag = True
    example = 'markowitz'

    # the location specified in the aggregation cfg file
    
    data_yaml_filename = f"{orig_cwd}/outputs/{example}/data_setup_outputs/{datetime0}/.hydra/config.yaml"

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}
    pen_ret = 10**setup_cfg['pen_rets_min']
    # extract M
    a = setup_cfg['a']
    static_dict = markowitz.static_canon(setup_cfg['data'], a, setup_cfg['idio_risk'], setup_cfg['scale_factor'])
    M = static_dict['M']
    def get_q(theta):
        q = jnp.zeros(2*a + 1)
        q = q.at[:a].set(-theta * pen_ret)
        q = q.at[a].set(1)
        return q
    get_q_batch = vmap(get_q, in_axes=(0), out_axes=(0))

    with open('data_setup_copied.yaml', 'w') as file:
        yaml.dump(setup_cfg, file)
    save_aggregate(static_flag, M, datetimes, example, get_q_batch)
    


@hydra.main(config_path='configs/osc_mass', config_name='osc_mass_agg.yaml')
def osc_mass_main(cfg):
    '''
    given data and time of all the previous experiments
    creates new folder with hydra
    combines all the npz files into 1
    '''
    # access first datafile via the data_cfg file
    example = 'osc_mass'
    orig_cwd = hydra.utils.get_original_cwd()

    datetimes = cfg.datetimes
    if len(datetimes) == 0:
        # get the most recent datetime and update datetimes
        last_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        datetimes = [last_datetime]

        cfg = {'datetimes': datetimes}
        
    # save the datetime to we can recover
    with open('agg_datetimes.yaml', 'w') as file:
        yaml.dump(cfg, file)

    datetime0 = datetimes[0]

    folder = f"{orig_cwd}/outputs/{example}/data_setup_outputs/{datetime0}"
    folder_entries = os.listdir(folder)
    entry = folder_entries[0]

    '''
    the next line is not right -- need to get the setup cfg file from
    '''
    #    the location specified in the aggregation cfg file
    data_yaml_filename = f"{orig_cwd}/outputs/{example}/data_setup_outputs/{datetime0}/.hydra/config.yaml"

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}
    T, control_box = setup_cfg['T'], setup_cfg['control_box']
    state_box = setup_cfg['state_box']
    nx, nu = setup_cfg['nx'], setup_cfg['nu']
    Q_val, QT_val = setup_cfg['Q_val'], setup_cfg['QT_val']
    R_val = setup_cfg['R_val']
    Ad, Bd = osc_mass.oscillating_masses_setup(nx, nu)

    # save the data setup cfg file in the aggregate folder
    with open('data_setup_copied.yaml', 'w') as file:
        yaml.dump(setup_cfg, file)

    static_dict = osc_mass.static_canon(T, nx, nu,
                                    state_box,
                                    control_box,
                                    Q_val,
                                    QT_val,
                                    R_val,
                                    Ad=Ad,
                                    Bd=Bd)
    A_sparse = static_dict['A_sparse']
    m, n = A_sparse.shape
    get_q_single = functools.partial(osc_mass.single_q,
                                        m=m,
                                        n=n,
                                        T=T,
                                        nx=nx,
                                        nu=nu,
                                        state_box=state_box,
                                        control_box=control_box,
                                        A_dynamics=Ad)
    get_q = vmap(get_q_single, in_axes=0, out_axes=0)
    M = static_dict['M']
    static_flag = True
    save_aggregate(static_flag, M, datetimes, example, get_q)


@hydra.main(config_path='configs/vehicle', config_name='vehicle_agg.yaml')
def vehicle_main(cfg):
    '''
    given data and time of all the previous experiments
    creates new folder with hydra
    combines all the npz files into 1
    '''
    example = 'vehicle'
    orig_cwd = hydra.utils.get_original_cwd()
    # access first datafile via the data_cfg file
    datetimes = cfg.datetimes
    if len(datetimes) == 0:
        # get the most recent datetime and update datetimes
        last_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        datetimes = [last_datetime]
        cfg = {'datetimes': datetimes}
    # save the datetime to we can recover
    with open('agg_datetimes.yaml', 'w') as file:
        yaml.dump(cfg, file)
    datetime0 = datetimes[0]
    example = 'vehicle'
    folder = f"{orig_cwd}/outputs/{example}/data_setup_outputs/{datetime0}"
    folder_entries = os.listdir(folder)
    entry = folder_entries[0]

    '''
    the next line is not right -- need to get the setup cfg file from
    '''
    #    the location specified in the aggregation cfg file
    data_yaml_filename = f"{orig_cwd}/outputs/{example}/data_setup_outputs/{datetime0}/.hydra/config.yaml"

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}
    nx, nu = 4, 3
    T, nx, nu, dt = setup_cfg['T'], 4, 3, setup_cfg['dt']
    state_box = np.inf
    control_box = np.array([setup_cfg['Fy_box'] * setup_cfg['Fy_factor'],
                            setup_cfg['Mx_box'] * setup_cfg['Mx_factor'],
                            setup_cfg['Mz_box'] * setup_cfg['Mz_factor']])
    Q_vec = np.array([setup_cfg['slip_penalty'], setup_cfg['yaw_penalty'],
                     setup_cfg['roll_penalty'], setup_cfg['roll_rate_penalty']])
    QT_vec = np.array([setup_cfg['slip_penalty'], setup_cfg['yaw_penalty'],
                      setup_cfg['roll_penalty'], setup_cfg['roll_rate_penalty']])
    R_vec = np.array([setup_cfg['Fy_penalty'] / (setup_cfg['Fy_factor'] ** 2),
                      setup_cfg['Mx_penalty'] / (setup_cfg['Mx_factor'] ** 2),
                      setup_cfg['Mz_penalty'] / (setup_cfg['Mz_factor'] ** 2)])
    delta_control_box = np.array([
        setup_cfg['control_rate_lim'] * setup_cfg['Fy_factor'],
        setup_cfg['control_rate_lim'] * setup_cfg['Mx_factor'],
        setup_cfg['control_rate_lim'] * setup_cfg['Mz_factor']
    ])

    # save the data setup cfg file in the aggregate folder
    # (so we no longer need the setup file)
    with open('data_setup_copied.yaml', 'w') as file:
        yaml.dump(setup_cfg, file)

    static_dict = vehicle.static_canon(T, nx, nu,
                                       state_box,
                                       control_box,
                                       Q_vec,
                                       QT_vec,
                                       R_vec,
                                       Ad=None,
                                       Bd=None,
                                       delta_control_box=delta_control_box)
    M_tilde = jnp.array(static_dict['M'])
    c_tilde, b_tilde = static_dict['c'], static_dict['b']
    q_tilde = jnp.concatenate([c_tilde, b_tilde])
    A_sparse = static_dict['A_sparse']
    m, n = A_sparse.shape
    '''
    canonicalize all at once with vmap
    '''
    get_M_q_single_partial = functools.partial(
        vehicle.get_single_M_q, M_tilde=M_tilde, q_tilde=q_tilde,
        T=T, m=m, n=n, dt=dt,
        u_box=jnp.array(control_box),
        Fy_factor=setup_cfg['Fy_factor'], Mx_factor=setup_cfg['Mx_factor'],
        Mz_factor=setup_cfg['Mz_factor'])
    get_M_q_batch = vmap(get_M_q_single_partial,
                         in_axes=(0,), out_axes=(0, 0,))
    static_flag = False
    M = None
    save_aggregate(static_flag, M, datetimes, example, get_M_q_batch)


def save_aggregate(static_flag, M, datetimes, example, get_q):
    if static_flag:
        @jit
        def get_w_star_q(x_star, y_star, q):
            u_star = jnp.hstack([x_star, y_star])
            return M @ u_star + u_star + q
        batch_get_w_star_q = vmap(
            get_w_star_q, in_axes=(0, 0, 0), out_axes=(0))
    else:
        @jit
        def get_w_star_Mq(x_star, y_star, M, q):
            u_star = jnp.hstack([x_star, y_star])
            return M @ u_star + u_star + q
        batch_get_w_star_Mq = vmap(
            get_w_star_Mq, in_axes=(0, 0, 0, 0), out_axes=(0))

    orig_cwd = hydra.utils.get_original_cwd()
    thetas_list = []
    q_mat_list = []
    x_stars_list = []
    y_stars_list = []
    matrix_invs_list = []
    M_tensor_list = []

    for datetime in datetimes:
        folder = f"{orig_cwd}/outputs/{example}/data_setup_outputs/{datetime}"
        folder_entries = os.listdir(folder)
        for entry in folder_entries:
            if 'data_setup_slurm' in entry:
                filename = f"{orig_cwd}/outputs/{example}/data_setup_outputs/{datetime}/{entry}"
                print(f"loading file {filename}")
                jnp_load_obj = jnp.load(filename)
                thetas = jnp_load_obj['thetas']

                if static_flag:
                    q_mat = get_q(thetas)
                else:
                    M_tensor, q_mat = get_q(thetas)
                    M_tensor_list.append(M_tensor)

                x_stars = jnp_load_obj['x_stars']
                y_stars = jnp_load_obj['y_stars']

                thetas_list.append(thetas)
                q_mat_list.append(q_mat)
                x_stars_list.append(x_stars)
                y_stars_list.append(y_stars)
        
    thetas = jnp.vstack(thetas_list)
    x_stars = jnp.vstack(x_stars_list)
    y_stars = jnp.vstack(y_stars_list)
    q_mat = jnp.vstack(q_mat_list)
    output_filename = 'data_setup_aggregate'

    if static_flag:
        w_stars = batch_get_w_star_q(x_stars, y_stars, q_mat)
        jnp.savez(output_filename,
                  thetas=thetas,
                  x_stars=x_stars,
                  y_stars=y_stars,
                  w_stars=w_stars
                  )
    else:
        M_tensors = jnp.vstack(M_tensor_list)
        m, n = y_stars.shape[1], x_stars.shape[1]

        @jit
        def inv(M_in):
            return jnp.linalg.inv(M_in + jnp.eye(m+n))
        batch_inv  = vmap(inv, in_axes=(0), out_axes=(0))
        print('inverting...')
        t0 = time.time()

        matrix_invs_tensor = batch_inv(M_tensors)
        
        t1 = time.time()
        print('inversion time', t1 - t0)
        print('M tensor shape', M_tensor.shape)
        print('q mat shape', q_mat.shape)
        
        M_tensor = jnp.vstack(M_tensor_list)
        w_stars = batch_get_w_star_Mq(x_stars, y_stars, M_tensor, q_mat)
        jnp.savez(output_filename,
                  thetas=thetas,
                  x_stars=x_stars,
                  y_stars=y_stars,
                  w_stars=w_stars,
                  matrix_invs=matrix_invs_tensor
                  )


if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/gpfs/rajivs/learn2warmstart/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'markowitz':
        # step 1. remove the markowitz argument -- otherwise hydra uses it as an override
        # step 2. add the train_outputs/... argument for train_outputs not outputs
        # sys.argv[1] = 'hydra.run.dir=outputs/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv[1] = base + \
            'markowitz/aggregate_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        markowitz_main()
    elif sys.argv[1] == 'osc_mass':
        # step 1. remove the markowitz argument -- otherwise hydra uses it as an override
        # step 2. add the train_outputs/... argument for train_outputs not outputs
        sys.argv[1] = base + \
            'osc_mass/aggregate_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        osc_mass_main()
    elif sys.argv[1] == 'vehicle':
        # step 1. remove the markowitz argument -- otherwise hydra uses it as an override
        # step 2. add the train_outputs/... argument for train_outputs not outputs
        sys.argv[1] = base + \
            'vehicle/aggregate_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        vehicle_main()
