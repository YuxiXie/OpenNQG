#!/bin/bash

set -x

DATAHOME=/storage/lmpan/xyx/OpenNQG/data/HotpotQA
EXEHOME=/storage/lmpan/xyx/OpenNQG/code

cd ${EXEHOME}

python preprocess.py \
       -train_src ${DATAHOME}/data/train/train.src.txt -train_tgt ${DATAHOME}/data/train/train.tgt.txt \
       -valid_src ${DATAHOME}/data/dev/dev.src.txt -valid_tgt ${DATAHOME}/data/dev/dev.tgt.txt \
       -answer enc -train_ans ${DATAHOME}/data/train/train.ans.txt -valid_ans ${DATAHOME}/data/dev/dev.ans.txt \
       -save_data ${DATAHOME}/data/rnn.enc.cp.data.pt \
       -src_seq_length 200 -tgt_seq_length 50 \
       -src_vocab_size 80000 -tgt_vocab_size 80000 \
       -src_words_min_frequency 4 -tgt_words_min_frequency 1 \
       -vocab_trunc_mode frequency \
       -pre_trained_vocab /storage/lmpan/xyx/word-vec/glove.6B.300d.txt -word_vec_size 300 \
       -copy \
       -batch_size 32
