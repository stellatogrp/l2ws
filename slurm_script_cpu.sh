#!/bin/bash
#SBATCH --job-name=array-job     # create a short name for your job
#SBATCH --output=slurm-%A.%a.out # STDOUT file
#SBATCH --error=slurm-%A.%a.err  # STDERR file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=160G         # memory per cpu-core (4G is default)
#SBATCH --array=0             # job array with index values 0, 1, 2, 3, 4
#SBATCH --time=01:39:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=rajivs@princeton.edu # 


echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

python l2ws_train_script.py sparse_pca cluster
# python gif_script.py robust_pca cluster
# python utils/portfolio_utils.py
# python plot_script.py mnist cluster
# python l2ws_setup_script.py robust_kalman cluster
#python scs_c_speed.py markowitz
# python aggregate_slurm_runs_script.py robust_pca cluster

# gpu command: #SBATCH --gres=gpu:1 