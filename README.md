# l2ws
This repository is by
[Rajiv Sambharya](https://rajivsambharya.github.io/),
[Georgina Hall](https://sites.google.com/view/georgina-hall),
[Brandon Amos](http://bamos.github.io/),
and [Bartolomeo Stellato](https://stellato.io/),
and contains the Python source code to
reproduce the experiments in our paper
"[Learning to Warm-Start Fixed-Point Optimization Algorithms]()."

If you find this repository helpful in your publications,
please consider citing our paper.

# Abstract
We introduce a machine-learning framework to warm-start fixed-point optimization algorithms. Our architecture consists of a neural network mapping problem parameters to warm starts, followed by a predefined number of fixed-point iterations. We propose two loss functions designed to either minimize the fixed-point residual or the distance to a ground truth solution. In this way, the neural network predicts warm starts with the end-to-end goal of minimizing the downstream loss. An important feature of our architecture is its flexibility, in that it can predict a warm start for fixed-point algorithms run for any number of steps, without being limited to the number of steps it has been trained on. We provide PAC- Bayes generalization bounds on unseen data for common classes of fixed-point operators: contractive, linearly convergent, and averaged. Applying this framework to well-known applications in control, statistics, and signal processing, we observe a significant reduction in the number of iterations and solution time required to solve these problems, through learned warm starts.

## Dependencies
Install dependencies with
```
pip install -r requirements.txt
```

## Instructions
### Running experiments
Experiments can from the root folder using the commands below.
The different experiments are
```
unconstrained_qp
lasso
quadcopter
mnist
robust_kalman
robust_ls
phase_retrieval
sparse_pca
```

```
python l2ws_setup.py quadcopter local
python aggregate_slurm_runs_script.py quadcopter local
python l2ws_train.py quadcopter local
python plot_script.py quadcopter local
```



We use the EMNIST dataset found at https://data.nasdaq.com/databases/WIKIP/documentation. To process the data run
```
python utils/portfolio_utils.py
```

To run our experiment run
```
python l2ws_setup.py markowitz local
python l2ws_train.py markowitz local
python plot_script.py markowitz local
```

Output folders will automatically be created from hydra and for the oscillating masses example, the plot and csv files to check the performance on different models will be creted in this file.
```
outputs/quadcopter/2022-12-03/14-54-32/plots/eval_iters.pdf
outputs/quadcopter/2022-12-03/14-54-32/plots/accuracies.csv
```

Adjust the config files to try different settings; for example, the number of train/test data, number of evaluation iterations, neural network training, and problem setup configurations. We automatically use the most recent output after each stage, but the specific datetime can be inputted. Additionally, the final evaluation plot can take in multiple training datetimes in a list. See the commented out lines in the config files.
