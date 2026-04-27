# L2WS
This repository is by
[Rajiv Sambharya](https://rajivsambharya.github.io/),
[Georgina Hall](https://sites.google.com/view/georgina-hall),
[Brandon Amos](http://bamos.github.io/),
and [Bartolomeo Stellato](https://stellato.io/),
and contains the Python source code to
reproduce the experiments in our paper
"[Learning to Warm-Start Fixed-Point Optimization Algorithms](https://arxiv.org/pdf/2309.07835.pdf)."
For an earlier conference version targeting QPs only, check out [this repo](https://github.com/stellatogrp/l2ws_qp).

If you find this repository helpful in your publications, please consider citing our papers.

# Abstract
We introduce a machine-learning framework to warm-start fixed-point optimization algorithms. Our architecture consists of a neural network mapping problem parameters to warm starts, followed by a predefined number of fixed-point iterations. We propose two loss functions designed to either minimize the fixed-point residual or the distance to a ground truth solution. In this way, the neural network predicts warm starts with the end-to-end goal of minimizing the downstream loss. An important feature of our architecture is its flexibility, in that it can predict a warm start for fixed-point algorithms run for any number of steps, without being limited to the number of steps it has been trained on. We provide PAC- Bayes generalization bounds on unseen data for common classes of fixed-point operators: contractive, linearly convergent, and averaged. Applying this framework to well-known applications in control, statistics, and signal processing, we observe a significant reduction in the number of iterations and solution time required to solve these problems, through learned warm starts.

## Installation

`l2ws` requires **Python 3.10**. The dependency stack is pinned to the versions
that match the paper's environment; in particular `jax==jaxlib==0.4.20`,
`optax==0.1.5`, and `jaxopt==0.6`. See the comments in `pyproject.toml` for
why each pin is required — loosening them breaks the codebase as written.

We recommend [`uv`](https://docs.astral.sh/uv/) to manage the virtual
environment because the pins lock to older versions that the default
resolver order can mishandle.

```bash
git clone https://github.com/stellatogrp/l2ws.git
cd l2ws
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

`pip` works too if you prefer:

```bash
git clone https://github.com/stellatogrp/l2ws.git
cd l2ws
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The install pulls a pinned commit of [`trajax`](https://github.com/google/trajax)
from GitHub, so a working network connection is required the first time.

The package is GPU-optional — JAX defaults to CPU, which is enough to run
every experiment (though training can take a long time without a GPU).

## Getting started

### Intro tutorials
You can find introductory tutorials on how to use `l2ws` in the folder `tutorials/`.


### Running experiments

From the `benchmarks/` folder:

```
python l2ws_setup.py <example> local
python l2ws_train.py <example> local
```

Replace `<example>` with one of:

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

The first script generates the problem instances and a training set of
optimal solutions; the second trains the warm-start network and writes the
per-iteration evaluation curves and (where applicable) closed-loop
rollouts. Both write to a fresh dated folder under
`outputs/<example>/{data_setup_outputs,train_outputs}/<date>/<time>/`.

A third script `plot.py` aggregates results across multiple training runs
to produce the paper figures; it is not needed to reproduce a single
experiment and is not described here.

***
#### ```l2ws_setup.py```

The first script ```l2ws_setup.py``` creates all of the problem instances and solves them.
The number of problems that are being solved is set in the setup config file.
That config file also includes other parameters that define the problem instances. 
This only needs to be run once for each example.
Depending on the example, this can take some time because 10000 problems are being solved.
After running this script, the results are saved a file in
```
outputs/quadcopter/data_setup_outputs/2022-06-03/14-54-32/
```

***
#### ```l2ws_train.py```

The second script ```l2ws_train.py``` does the actual training using the output from the prevous setup command.
In particular, in the config file, it takes a datetime that points to the setup output.
By default, it takes the most recent setup if this pointer is empty.
The train config file holds information about the actual training process.
Run this file for each $k$ value to train for that number of fixed-point steps.
Each run for a given $k$ and the loss function creates an output folder like
To replicate our results in the paper, the only inputs that need to be changed are the ones that determine the number of training steps and which of the two loss functions you are using.
- ```train_unrolls``` (an integer that is the value $k$)
- ```supervised``` (either True or False)

```
outputs/quadcopter/train_outputs/2022-06-04/15-14-05/
```
In this folder there are many metrics that are stored.
We highlight the mains ones here (both the raw data in csv files and the corresponding plots in pdf files).


- Fixed-point residuals over the test problems 

    ```outputs/quadcopter/train_outputs/2022-06-04/15-14-05/plots/iters_compared_test.csv```
    ```outputs/quadcopter/train_outputs/2022-06-04/15-14-05/plots/eval_iters_test.pdf```

- Fixed-point residuals over the training problems 

    ```outputs/quadcopter/train_outputs/2022-06-04/15-14-05/plots/iters_compared_train.csv```
    ```outputs/quadcopter/train_outputs/2022-06-04/15-14-05/eval_iters_train.pdf```

- Losses over epochs: for training this holds the average loss (for either loss function), for testing we plot the fixed-point residual at $k$ steps

    ```outputs/quadcopter/train_outputs/2022-06-04/15-14-05/train_test_results.csv```
    ```outputs/quadcopter/train_outputs/2022-06-04/15-14-05/losses_over_training.pdf```

- The ```accuracies``` folder holds the results that are used for the tables. First, it holds the average number of iterations to reach the desired accuracies ($0.1$, $0.01$, $0.001$, and $0.0001$ by default).
Second, it holds the reduction in iterations in comparison to the cold start.

    ```outputs/quadcopter/2022-12-03/14-54-32/plots/accuracies```

- The ```solve_c``` folder holds the results that we show for the timings (for OSQP and SCS) in the paper.
In the train config file, we can set the accuracies that we set OSQP and SCS to (both the relative and absolute accuracies are set to the same value).

    ```outputs/quadcopter/2022-12-03/14-54-32/plots/solve_c```



***
#### ```plot.py```

The third script ```plot.py``` plots the results across many different training runs.
Each train run creates a new folder 
```
outputs/quadcopter/plots/2022-06-04/15-14-05/
```



### Quadcopter side-by-side comparison GIF

After training the quadcopter example, you can produce a slide-ready,
multi-pane GIF that compares closed-loop tracking under different
warm-start strategies on the same reference trajectory. Two scripts
live in `benchmarks/scripts/`:

```
python benchmarks/scripts/rerun_quadcopter_rollouts.py
python benchmarks/scripts/restyle_quadcopter_gif.py
```

`rerun_quadcopter_rollouts.py` re-runs only the closed-loop rollouts
(no retraining) using the saved NN weights from the most recent
training run. By default it picks up:

- the most recent `benchmarks/outputs/quadcopter/data_setup_outputs/<DATE>/<TIME>/`
- the most recent `benchmarks/outputs/quadcopter/train_outputs/<DATE>/<TIME>/`,
  which must contain `nn_weights/layer_*_params.npz` (these are saved when
  `save_weights_flag: true` in `quadcopter_run.yaml`, the default)

Use `--setup-datetime` and `--train-datetime` to point at a different pair.
The script runs four rollouts for each of `nearest_neighbor`, `prev_sol`,
and `learned`, writing per-rollout state arrays to
`benchmarks/outputs/quadcopter/restyle_rollouts/<DATE>/<TIME>/rollouts/{method}/rollout_{i}_states.npz`.
On a Mac M4 CPU the bulk of the time is the one-off batch factorisation
of the 11k KKT systems (a few minutes); the rollouts themselves are sub-minute.

`restyle_quadcopter_gif.py` consumes those state arrays and renders a
paper/talk-quality animation: a stylised quadcopter mesh (body + four
spinning rotors), fading position trail, ground shadow that matches
the body+rotor footprint, dashed reference trajectory, and a gently
orbiting camera shared across panes. Defaults render rollout `#3` as a
3-pane vertical comparison (`nearest_neighbor`, `prev_sol`, `learned`)
and write the GIF next to the rollouts directory:

```
benchmarks/outputs/quadcopter/restyle_rollouts/<DATE>/<TIME>/rollout_3_compare.gif
```

Useful flags:

- `--rollout-index <i>` — pick which of the four rollouts to render
- `--methods <a> <b> ...` — choose a subset / different ordering of
  method subdirs; defaults to `nearest_neighbor prev_sol learned`
- `--layout {vertical,horizontal}` — vertical (default) suits a slide
  column; horizontal suits a wide slide
- `--rollouts-dir <path>` — point at a specific `rollouts/` directory
  instead of the most recent one
- `--output <path>` — override the output GIF location

The renderer module itself lives at `l2ws/examples/quadcopter_render.py`
and exposes `make_compare_gif(...)` if you want to call it from your
own driver.

For the image deblurring task, we use the EMNIST dataset found at https://www.nist.gov/itl/products-and-services/emnist-dataset and use pip to install emnist (https://pypi.org/project/emnist/). 


Adjust the config files to try different settings; for example, the number of train/test data, number of evaluation iterations, and the number of training steps.
Additionally, the neural network and problem setup configurations can be updated.
We automatically use the most recent output after each stage, but the specific datetime can be inputted. Additionally, the final evaluation plot can take in multiple training datetimes in a list. See the commented out lines in the config files.

***


# Important files in the backend
To reproduce our results, this part is not needed.

- The ```l2ws/examples``` folder holds the code for each of the numerical experiments we run. The main purpose is to be used in conjunction with the ```l2ws_setup.py```.

- An important note is that the code is set to periodically evaluate the train and test sets; this is set in the ```eval_every_x_epochs``` entry in the run config file.
When we evaluate, the fixed-point curves are updated (see the above files for the run config).

We can also set the number of problems we run with C (for OSQP and SCS) with ```solve_c_num```. This will create the results that are used for our timing tables.
***

The ```l2ws``` folder holds the code that implements our architecture and allows for the training. In particular,

- ```l2ws/launcher.py``` is the workspace which holds the L2WSmodel below.
All of the evaluation and training is run through

- ```l2ws/algo_steps.py``` holds all of the code that runs the algorithms

    - the fixed-point algorithms follow the same form in case you want to try your own algorithm

- ```l2ws/l2ws_model.py``` holds the L2WSmodel object, i.e., the architecture. This code allows us to 
    - evaluate the problems (both test and train) for any initialization technique
    - train the neural network weights with the given parameters: the number of fixed-point steps in the architecture $k$ (```train_unrolls```) and the training loss $`\ell^{\rm fp}_{\theta}`$ (```supervised=False```) or $`\ell^{\rm reg}_{\theta}`$ (```supervised=True```)
