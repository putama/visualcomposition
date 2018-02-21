#!/bin/bash
#
#  Execute from the current working directory
#$ -cwd
#
#  This is a long-running job
#$ -l gpus=1
#
#$ -o gridlogs
#
#$ -e gridlogs

source /data/people/putama/pytorch_venv/bin/activate
echo "=> job submitted"

python -u $@
