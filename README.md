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
Experiments can from the root folder using the commands below for the quadcopter example.
```
python l2ws_setup.py quadcopter local
python l2ws_train.py quadcopter local
python plot_script.py quadcopter local
```
The first script ```l2ws_setup.py``` creates all of the problem instances and solves them.
The number of problems that are being solved is set in the setup config file.
That config file also includes other parameters that define the problem instances. 
This only needs to be run once for each example.
Depending on the example, this can take some time because 10000 problems are being solved.
After running this script, the results are saved a file in
```
outputs/quadcopter/data_setup_outputs/2022-06-03/14-54-32/
```

The second script ```l2ws_train.py``` does the actual training.
The train config file holds information about the actual training process.
Run this file for each $k$ value to train for that number of fixed-point steps.
Each run for a given $k$ and the loss function creates an output folder like
```
outputs/quadcopter/train_outputs/2022-06-04/15-14-05/
```
In this folder there are many metrics that are stored.
We highlight the mains ones here.


- Fixed-point residuals over the test problems 

    ```outputs/quadcopter/train_outputs/2022-12-03/14-54-32/plots/iters_compared_test.pdf```
    ```outputs/quadcopter/train_outputs/2022-12-03/14-54-32/plots/eval_iters_test.pdf```

- Fixed-point residuals over the training problems 

    ```outputs/quadcopter/train_outputs/2022-12-03/14-54-32/plots/iters_compared_train.pdf```
    ```outputs/quadcopter/train_outputs/2022-12-03/14-54-32/eval_iters_train.pdf```

- Losses over epochs: for training this holds the average loss (for either loss function), for testing we plot the fixed-point residual at $k$ steps

    ```outputs/quadcopter/train_outputs/2022-12-03/14-54-32/plots/losses_over_training.csv```
    ```outputs/quadcopter/train_outputs/2022-12-03/14-54-32/plots/losses_over_training.pdf```



The third script ```plot_script.py``` plots the results across many different training runs.
Each train run creates a new folder 
```
outputs/quadcopter/plots/2022-06-04/15-14-05/
```

The names of the different experiments are the following.
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

For the image deblurring task, we use the EMNIST dataset found at https://www.nist.gov/itl/products-and-services/emnist-dataset and use pip to install emnist (https://pypi.org/project/emnist/). 

Output folders will automatically be created from hydra and for the oscillating masses example, the plot and csv files to check the performance on different models will be creted in this file.
```
outputs/quadcopter/2022-12-03/14-54-32/plots/eval_iters_test.pdf
outputs/quadcopter/2022-12-03/14-54-32/plots/eval_iters_train.pdf
outputs/quadcopter/2022-12-03/14-54-32/plots/losses_over_training.pdf
```
The csv files for the fixed-point residuals over the evaluation steps exist in
```
outputs/quadcopter/2022-12-03/14-54-32/plots/accuracies
```

Adjust the config files to try different settings; for example, the number of train/test data, number of evaluation iterations, and the number of training steps.
Additionally, the neural network and problem setup configurations can be updated.
We automatically use the most recent output after each stage, but the specific datetime can be inputted. Additionally, the final evaluation plot can take in multiple training datetimes in a list. See the commented out lines in the config files.
