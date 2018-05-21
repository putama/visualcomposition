#!/bin/bash
#
#  Execute from the current working directory
#$ -cwd
#
#  This is a long-running job
#$ -l gpus=4
#
#$ -o gridlogs
#
#$ -e gridlogs

source /data/people/putama/pytorch_venv/bin/activate
echo "=> job submitted"

python vse_train.py --data_path data \
    --data_name coco --which_vocab cocomit --logger_name runs/cocomit_vse++_vgg_finetune \
    --max_violation --batch_size 128 --use_restval --finetune --cnn_type vgg19
