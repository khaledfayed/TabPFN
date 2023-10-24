#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition aisdlc_gpu-rtx2080    # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name metanet             # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output logs/%x-%A-HelloCluster.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x-%A-HelloCluster.err    # STDERR  short: -e logs/%x-%A-job_name.out

# CPU settings
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks=1
#SBATCH --nodes=1

# Define the amount of memory required per node
#SBATCH --mem 8GB

#SBATCH --gres=gpu:1

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
source ~/.bashrc # Adjust to your path of Miniconda installation
conda activate thesis

# Running the job
start=`date +%s`

python train_new_try_meta.py --epochs 4000 --lr 0.0001 --weight_decay 0.0001 --name "1.shuffle"

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime