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

python vse_train.py --data_path data \
    --data_name coco --logger_name runs/coco_vse++_restval \
    --max_violation --batch_size 128 --use_restval --which_vocab coco
