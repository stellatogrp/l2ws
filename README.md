# l2ws
This repository is by
[Rajiv Sambharya](https://rajivsambharya.github.io/),
[Georgina Hall](https://sites.google.com/view/georgina-hall),
[Brandon Amos](http://bamos.github.io/),
and [Bartolomeo Stellato](https://stellato.io/),
and contains the Python source code to
reproduce the experiments in our paper
"[End-to-End Learning to Warm-Start for Real-Time Quadratic Optimization]()."

If you find this repository helpful in your publications,
please consider citing our paper.

# Abstract
First-order methods are widely used to solve convex quadratic programs (QPs) in real-time applications because of their low per-iteration cost. 
However, they can suffer from slow convergence to accurate solutions. 
In this paper, we present a framework which learns an effective warm-start for a popular first-order method in real-time applications, Douglas-Rachford (DR) splitting, across a
family of parametric QPs. 
This framework consists of two modules: a feedforward neural network
block, which takes as input the parameters of the QP and outputs a warm-start, and a block which
performs a fixed number of iterations of DR splitting from this warm-start and outputs a candidate
solution. 
A key feature of our framework is its ability to do end-to-end learning as we differentiate through the DR iterations. 
To illustrate the effectiveness of our method, we provide generalization
bounds (based on Rademacher complexity) that improve with the number of training problems and
number of iterations simultaneously. 
We further apply our method to three real-time applications
and observe that, by learning good warm-starts, we are able to significantly reduce the number of
iterations required to obtain high-quality solutions.

## Dependencies
Install dependencies with
```
pip install -r requirements.txt
```

## Instructions
### Running experiments
Experiments can from the root folder using the commands below.

Oscillating masses:
```
python l2ws_setup.py osc_mass local
python aggregate_slurm_runs_script.py osc_mass local
python l2ws_train.py osc_mass local
python plot_script.py osc_mass local
```
Vehicle dynamics:
```
python l2ws_setup.py vehicle local
python aggregate_slurm_runs_script.py vehicle local
python l2ws_train.py vehicle local
python plot_script.py vehicle local
```
Markowitz:
To get the data, from NASDAQ you must create an account with NASDAQ (https://data.nasdaq.com/) and download the data with 
```
https://data.nasdaq.com/tables/WIKI-PRICES/export?api_key={INSERT_API_KEY}[…]date&qopts.columns%5B%5D=ticker&qopts.columns%5B%5D=adj_close
```

We use the WIKIPRICES dataset found at https://data.nasdaq.com/databases/WIKIP/documentation. To process the data run
```
python utils/portfolio_utils.py
```

To run our experiment run
```
python l2ws_setup.py markowitz local
python aggregate_slurm_runs_script.py markowitz local
python l2ws_train.py markowitz local
python plot_script.py markowitz local
```

Output folders will automatically be created from hydra and for the oscillating masses example, the plot and csv files to check the performance on different models will be creted in this file.
```
outputs/osc_mass/2022-12-03/14-54-32/plots/eval_iters.pdf
outputs/osc_mass/2022-12-03/14-54-32/plots/accuracies.csv
```

Adjust the config files to try different settings; for example, the number of train/test data, number of evaluation iterations, neural network training, and problem setup configurations. We automatically use the most recent output after each stage, but the specific datetime can be inputted. Additionally, the final evaluation plot can take in multiple training datetimes in a list. See the commented out lines in the config files.
