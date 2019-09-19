#!/bin/bash

set -x

DATAHOME=/storage/lmpan/xyx/OpenNQG/data/HotpotQA
EXEHOME=/storage/lmpan/xyx/OpenNQG/code
MODELHOME=${DATAHOME}/models/bert-rnn/base
LOGHOME=${DATAHOME}/logs/bert-rnn/base

mkdir -p ${MODELHOME}
mkdir -p ${LOGHOME}

cd ${EXEHOME}

python train.py \
       -data ${DATAHOME}/data/bert.all.data.pt \
       -epoch 100 -batch_size 32 -eval_batch_size 16 \
       -answer sep \
       -max_token_src_len 200 -max_token_tgt_len 50 \
       -pretrained bert-base-uncased \
       -d_word_vec 300 \
       -d_dec_model 512 -n_dec_layer 1 -dec_rnn gru -d_k 64 \
       -copy -coverage -layer_attn \
       -maxout_pool_size 2 -n_warmup_steps 10000 \
       -dropout 0.5 -attn_dropout 0.1 \
       -gpus 0 \
       -save_mode best -save_model ${MODELHOME}/hotpotqa.model.bert-rnn.dfinal \
       -logfile_train ${LOGHOME}/hotpotqa.log.train.bert-rnn.dfinal \
       -logfile_dev ${LOGHOME}/hotpotqa.log.dev.bert-rnn.dfinal \
       -log_home ${LOGHOME} \
       -translate_ppl 15 \
       -curriculum 0  -extra_shuffle -optim adam -learning_rate 0.001 -learning_rate_decay 0.75 \
       -valid_steps 500 -decay_steps 500 -start_decay_steps 5000 -decay_bad_cnt 5 -max_grad_norm 5 -max_weight_value 20