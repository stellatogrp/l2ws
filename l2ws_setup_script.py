import sys
import examples.markowitz as markowitz
import examples.osc_mass as osc_mass
import examples.vehicle as vehicle
import hydra
import pdb
import cvxpy as cp
import scipy
import numpy as np


@hydra.main(config_path='configs/markowitz', config_name='markowitz_setup.yaml')
def main_setup_markowitz(cfg):
    markowitz.setup_probs(cfg)

    
@hydra.main(config_path='configs/osc_mass', config_name='osc_mass_setup.yaml')
def main_setup_osc_mass(cfg):
    osc_mass.setup_probs(cfg)


@hydra.main(config_path='configs/vehicle', config_name='vehicle_setup.yaml')
def main_setup_vehicle(cfg):
    vehicle.setup_probs(cfg)


if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/gpfs/rajivs/learn2warmstart/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'markowitz':
        # step 1. remove the markowitz argument -- otherwise hydra uses it as an override
        # step 2. add the train_outputs/... argument for data_setup_outputs not outputs
        # sys.argv[1] = 'hydra.run.dir=outputs/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv[1] = base + 'markowitz/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_markowitz()
    elif sys.argv[1] == 'osc_mass':
        sys.argv[1] = base + 'osc_mass/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_osc_mass()
    elif sys.argv[1] == 'vehicle':
        sys.argv[1] = base + 'vehicle/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_vehicle()
