#!/bin/bash

set -x

DATAHOME=/home/wing.nus/lmpan/OpenNQG/data/HotpotQA
# DATAHOME=/home/wing.nus/lmpan/OpenNQG/data/SQuAD
EXEHOME=/home/wing.nus/lmpan/OpenNQG/code

cd ${EXEHOME}

#####========== Pretrain: Language Model ==========#####
### fine-tune BERT using data pregenerated traning data —— input raw data - corpus
# python finetune_lm.py \
#        --train_file ${DATAHOME}/corpus/corpus_for_pretrain_LM_min.txt \
#        --bert_model bert-base-uncased \
#        --output_dir ${DATAHOME}/models/bert-LM/ \
#        --max_seq_len 256 --do_train --num_train_epochs 5 \
#        --gpus 3 5 4 6 8
### fine-tune BERT using data pregenerated traning data —— input raw data - corpus
python finetune_ner.py \
       -pre_model bert-base-uncased \
       -train_src ${DATAHOME}/train/train.src.txt -train_tgt ${DATAHOME}/train/train.ner.txt \
       -valid_src ${DATAHOME}/dev/dev.src.txt -valid_tgt ${DATAHOME}/dev/dev.ner.txt \
       -output_dir ${DATAHOME}/pretrain/models/bert-ner/ \
       -gpus 2



# {
#     "name": "nus3",
#     "host": "weisshorn.d2.comp.nus.edu.sg",
#     "protocol": "sftp",
#     "port": 22,
#     "username": "wing.nus",
#     "password": "y%bWvNw^",
#     "remotePath": "/home/wing.nus/lmpan/OpenNQG/code/",
#     "uploadOnSave": true
# }

# {
#     "name": "nus2",
#     "host": "next-gpu2.d2.comp.nus.edu.sg",
#     "protocol": "sftp",
#     "port": 22,
#     "username": "lmpan",
#     "password": "123@nus",
#     "remotePath": "/storage/lmpan/xyx/OpenNQG/code/",
#     "uploadOnSave": true
# }

