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
To install the package, run
```
$ pip install git+https://github.com/stellatogrp/l2ws
```

## Getting started

### Intro tutorials
You can find introductory tutorials on how to use `l2ws` in the folder `tutorials/`.


### Running experiments
To download the experiments, you should clone this repository with
```
git clone https://github.com/stellatogrp/l2ws_fixed_point.git
```
Experiments can from the `benchmarks/` folder using the commands below:
```
python l2ws_setup.py <example> local
python l2ws_train.py <example> local
python plot_script.py <example> local
```

Replace the ```<example> ``` with one of the following to run an experiment.
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
