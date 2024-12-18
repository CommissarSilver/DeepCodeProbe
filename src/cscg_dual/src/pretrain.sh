#!/bin/sh

L1='nl'
L2='code'
JOB='pretrain'

data_dir="/store/travail/vamaj/Leto/src/cscg_dual/dataset/python/original"
vocab_bin="$data_dir/vocab.bin"
train_src="$data_dir/train.token.${L1}"
train_tgt="$data_dir/train.token.${L2}"
test_src="$data_dir/valid.token.${L1}"
test_tgt="$data_dir/valid.token.${L2}"

job_name="$JOB"
model_name="/store/travail/vamaj/Leto/src/cscg_dual/pretrain_models/nl2c.${job_name}"

python nmt/nmt.py \
    --cuda \
    --gpu 0 \
    --load_model /store/travail/vamaj/Leto/src/cscg_dual/pretrain_models/nl2c.pretrain.iter12000.bin \
    --mode train \
    --vocab ${vocab_bin} \
    --save_to ${model_name} \
    --log_every 500 \
    --valid_niter 500 \
    --valid_metric bleu \
    --save_model_after 2 \
    --beam_size 10 \
    --batch_size 4 \
    --hidden_size 512 \
    --embed_size 512 \
    --uniform_init 0.1 \
    --dropout 0.2 \
    --clip_grad 5.0 \
    --decode_max_time_step 50 \
    --lr_decay 0.8 \
    --lr 0.002 \
    --train_src ${train_src} \
    --train_tgt ${train_tgt} \
    --dev_src ${test_src} \
    --dev_tgt ${test_tgt}

