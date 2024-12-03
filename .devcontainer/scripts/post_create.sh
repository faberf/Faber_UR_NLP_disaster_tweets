#!/bin/bash
# This script runs inside the container after it has been created.

echo "Post-create script running..."

# Activate the Conda environment (if needed)
source /opt/conda/etc/profile.d/conda.sh
conda activate dev_env

# Perform any additional setup here
