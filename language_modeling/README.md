# Language Modelling with Recurrent Fast Weight Programmers

This repository was originally forked from https://github.com/IDSIA/lmtool-fwp.

**Please refer to the original repository for issues and latest development of this toolkit.**

## Requirement
* PyTorch (PyT = 1.4.0)
* Ninja (`pip install ninja`)
* Optionally wandb for monitoring jobs (for disable it by removing the `--use_wandb` flag).

## Training
We run the following script to train models. 
```
    python train.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
        --n_layer 16 \
        --d_model 128 \
        --n_head 8 \
        --d_head 16 \
        --d_inner 2048 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 2000 \
        --max_step 500000 \
        --attn_type 134 \
        --tgt_len 256 \
        --mem_len 0 \
        --pre_lnorm \
        --skip_attn_normalization \
        --eval_tgt_len 256 \
        --batch_size 96 \
        --multi_gpu \
        --use_wandb \
        --project_name 'my_project'
```

In addition, for the Delta MLP model, we have an extra flag `--fast_net_depth` to specify the depth of the fast MLP (See Appendix A).
We gave different IDs for models we propose which can be specified via `--attn_type`.

The corresponding mapping is as follows:
* 54: Delta MLP (Table 2, and 4 in Appendix A)
* 64: Delta Delta Net (Table 2)
* 124: Delta RNN version 1 (Table 4 Appendix A)
* 134: Delta RNN version 2 (Table 2, and 4 in Appendix A)
* 224: Delta LSTM version 1 (Table 4 Appendix A)
* 234: Delta LSTM version 2 (Table 4 Appendix A)
* 244: Delta LSTM version 3 (Table 4 Appendix A)
* 254: Delta LSTM version 4 (Table 2, and 4 in Appendix A)
* 934: Recurrent Delta Net (Table 2)


## Evaluation
The evaluation script we used is as follows: 
```
    python eval_sliding_window.py \
        --cuda \
        --batch_size 1 \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 256 \
        --mem_len 0 \
        --clamp_len 256 \
        --split valid
```
By changing `valid` to `test` in the `--split` flag, we can compute test perplexity.

## Full context training and evaluation
For models trained and evaluated without context truncation ("+ full context" in Table 2),
the training script is as follows:
```
    python train.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
        --n_layer 16 \
        --d_model 128 \
        --n_head 8 \
        --d_head 16 \
        --d_inner 2048 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 2000 \
        --max_step 500000 \
        --attn_type 134 \
        --tgt_len 256 \
        --mem_len 0 \
        --eval_tgt_len 256 \
        --batch_size 96 \
        --skip_attn_normalization \
        --use_wandb \
        --no_pos \
        --carry_over_fast_weight \
        --project_name 'my_project'
```
and for evaluation:
```
    python eval.py \
        --cuda \
        --carry_over_fast_weight \
        --batch_size 1 \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 256 \
        --mem_len 0 \
        --clamp_len 256 \
        --split valid
```
