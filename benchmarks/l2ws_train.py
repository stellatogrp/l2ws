import sys
import l2ws.examples.markowitz as markowitz
import l2ws.examples.osc_mass as osc_mass
import l2ws.examples.vehicle as vehicle
import l2ws.examples.robust_kalman as robust_kalman
import l2ws.examples.robust_pca as robust_pca
import l2ws.examples.robust_ls as robust_ls
import l2ws.examples.sparse_pca as sparse_pca
import l2ws.examples.phase_retrieval as phase_retrieval
import l2ws.examples.lasso as lasso
import l2ws.examples.unconstrained_qp as unconstrained_qp
import l2ws.examples.mpc as mpc
import l2ws.examples.quadcopter as quadcopter
import l2ws.examples.mnist as mnist
import l2ws.examples.jamming as jamming
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


@hydra.main(config_path='configs/lasso', config_name='lasso_run.yaml')
def main_run_lasso(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'lasso'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    lasso.run(cfg)


@hydra.main(config_path='configs/quadcopter', config_name='quadcopter_run.yaml')
def main_run_quadcopter(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'quadcopter'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    quadcopter.run(cfg)


@hydra.main(config_path='configs/jamming', config_name='jamming_run.yaml')
def main_run_jamming(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'jamming'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    jamming.run(cfg)


@hydra.main(config_path='configs/mnist', config_name='mnist_run.yaml')
def main_run_mnist(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'mnist'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    mnist.run(cfg)


@hydra.main(config_path='configs/unconstrained_qp', config_name='unconstrained_qp_run.yaml')
def main_run_unconstrained_qp(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'unconstrained_qp'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    unconstrained_qp.run(cfg)


@hydra.main(config_path='configs/mpc', config_name='mpc_run.yaml')
def main_run_mpc(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'mpc'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    mpc.run(cfg)


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


@hydra.main(config_path='configs/sparse_pca', config_name='sparse_pca_run.yaml')
def main_run_sparse_pca(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'sparse_pca'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    sparse_pca.run(cfg)


@hydra.main(config_path='configs/phase_retrieval', config_name='phase_retrieval_run.yaml')
def main_run_phase_retrieval(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'phase_retrieval'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    phase_retrieval.run(cfg)


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
    elif sys.argv[1] == 'sparse_pca':
        sys.argv[1] = base + 'sparse_pca/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_sparse_pca()
    elif sys.argv[1] == 'phase_retrieval':
        sys.argv[1] = base + 'phase_retrieval/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_phase_retrieval()
    elif sys.argv[1] == 'lasso':
        sys.argv[1] = base + 'lasso/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_lasso()
    elif sys.argv[1] == 'mpc':
        sys.argv[1] = base + 'mpc/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_mpc()
    elif sys.argv[1] == 'unconstrained_qp':
        sys.argv[1] = base + 'unconstrained_qp/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_unconstrained_qp()
    elif sys.argv[1] == 'quadcopter':
        sys.argv[1] = base + 'quadcopter/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_quadcopter()
    elif sys.argv[1] == 'mnist':
        sys.argv[1] = base + 'mnist/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_mnist()
    elif sys.argv[1] == 'jamming':
        sys.argv[1] = base + 'jamming/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_jamming()
