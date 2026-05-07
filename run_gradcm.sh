#!/bin/bash
#SBATCH --job-name=ig-explanation
#SBATCH --account=ml4h_jobs
#SBATCH --time=00:30:00
#SBATCH --output=ig_results-%j.out
#SBATCH --error=ig_results-%j.err

set -euo pipefail

# 1. Load the cluster's GPU drivers
module load cuda/12.6

# 2. Activate the course's official virtual environment
source /cluster/courses/ml4h/jupyter/bin/activate

# 3. Navigate to your project folder
cd ~/ml4h_project
python grad_cam.py