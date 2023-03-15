import sys
import examples.markowitz as markowitz
import examples.osc_mass as osc_mass
import examples.vehicle as vehicle
import examples.robust_kalman as robust_kalman
import examples.robust_pca as robust_pca
import examples.robust_ls as robust_ls
import hydra
import pdb
import yaml
from l2ws.utils.data_utils import copy_data_file, recover_last_datetime


@hydra.main(config_path='configs/markowitz', config_name='markowitz_run.yaml')
def main_run_markowitz(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'markowitz'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'aggregate')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    markowitz.run(cfg)


@hydra.main(config_path='configs/osc_mass', config_name='osc_mass_run.yaml')
def main_run_osc_mass(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'osc_mass'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'aggregate')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    osc_mass.run(cfg)


@hydra.main(config_path='configs/robust_kalman', config_name='robust_kalman_run.yaml')
def main_run_robust_kalman(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'robust_kalman'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    robust_kalman.run(cfg)


@hydra.main(config_path='configs/robust_pca', config_name='robust_pca_run.yaml')
def main_run_robust_pca(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'robust_pca'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'aggregate')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    robust_pca.run(cfg)


@hydra.main(config_path='configs/robust_ls', config_name='robust_ls_run.yaml')
def main_run_robust_ls(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'robust_ls'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    robust_ls.run(cfg)


@hydra.main(config_path='configs/vehicle', config_name='vehicle_run.yaml')
def main_run_vehicle(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'vehicle'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'aggregate')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    vehicle.run(cfg)


if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/gpfs/rajivs/learn2warmstart/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'markowitz':
        # step 1. remove the markowitz argument -- otherwise hydra uses it as an override
        # step 2. add the train_outputs/... argument for train_outputs not outputs
        # sys.argv[1] = 'hydra.run.dir=outputs/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv[1] = base + 'markowitz/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_markowitz()
    elif sys.argv[1] == 'osc_mass':
        sys.argv[1] = base + 'osc_mass/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_osc_mass()
    elif sys.argv[1] == 'vehicle':
        sys.argv[1] = base + 'vehicle/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_vehicle()
    elif sys.argv[1] == 'robust_kalman':
        sys.argv[1] = base + 'robust_kalman/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_robust_kalman()
    elif sys.argv[1] == 'robust_pca':
        sys.argv[1] = base + 'robust_pca/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_robust_pca()
    elif sys.argv[1] == 'robust_ls':
        sys.argv[1] = base + 'robust_ls/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_robust_ls()
