#!/bin/bash

## YOU'LL HAVE TO CHANGE THESE SETTINGS FOR YOUR SLURM CONFIG.

#SBATCH --job-name=quick_llama_singlenode_example
#SBATCH --output=/scratch/${USER}/quick_llama_output_%j.out
#SBATCH --error=/scratch/${USER}/quick_llama_error_%j.err
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:2 # Request 2 GPUs on this single node

##################################################
# 1) Prepare quick_llama
##################################################
# Make sure it is accessible
# cd /scratch/${USER}
# git clone https://github.com/samblouir/quick_llama.git

##################################################
# Configure this to the directory you cloned quick_llama to:
##################################################
quick_llama_dir_path="/scratch/${USER}/quick_llama"

##################################################
# Load in modules
##################################################
ml gnu12 cuda cudnn nvidia-hpc-sdk git

##################################################
# Debugging
##################################################
echo "--- Debugging for Setup ---"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Allocated Nodes: $SLURM_JOB_NODELIST"
echo "Number of Tasks: $SLURM_NTASKS"
echo "GPUs Requested: 2"
echo "-----------------"


# Navigate to the working directory (exit if it fails)
cd ${quick_llama_dir_path} || { echo "FATAL ERROR; exiting. Could not navigate to ${quick_llama_dir_path}"; exit 1; }
cd src/quick_llama/minimal_training_example.py


##################################################
# Start
##################################################
echo "Starting..."

accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    minimal_training_example.py

##################################################
echo "quick_llama finished with exit code $?."
##################################################
