# Reinforcement Learning with Recurrent FWPs

This repository was originally forked from *Torchbeast* https://github.com/facebookresearch/torchbeast.

This is the codebase we used to train Recurrent Fast Weight Programmers and related models in Atari 2600 games.

## Requirements
* We use the `Polybeast` version of Torchbeast. To install it, we refer to the original instructions given at https://github.com/facebookresearch/torchbeast.
This might not be straightforward depending on the system.
For example, we had to apply a specific patch (file `install_grpc.patch`) before installing gRPC in our system:
```
patch -p1 < install_grpc.patch
```
Also, after installing, `libtorchbeast` directory should be renamed e.g. to `_libtorchbeast` so that you can call `import libtorchbeast`.

In addition to the `Polybeast` related requirements, we use:
* Ninja to compile our custom CUDA kernels (`pip install ninja`).
* Optionally, wandb to monitor our jobs (this can be disabled by removing `--use_wandb`; see example below).

## Training
An example training script is shown as follows:
```
export TORCH_EXTENSIONS_DIR="/home/me/torch_extensions/torchbeast"
export CUDA_VISIBLE_DEVICES=2,3

SAVE_DIR=saved_models

python -m torchbeast.polybeast \
     --env GopherNoFrameskip-v4 \
     --num_actors 48 \
     --num_servers 48 \
     --total_steps 200000000 \
     --save_extra_checkpoint 50000000 \
     --learning_rate 0.0006 \
     --grad_norm_clipping 40 \
     --epsilon 0.01 \
     --entropy_cost 0.01 \
     --batch_size 32 \
     --unroll_length 50 \
     --num_actions 8 \
     --use_rec_delta \
     --num_layers 2 \
     --use_wandb \
     --num_learner_threads 1 \
     --num_inference_threads 1 \
     --project_name "my_project" \
     --xpid "gopher_rec_delta" \
     --savedir ${SAVE_DIR}
```

To specify the model to be used, provide one of the following flags:
* No flag for the feedforward baseline
* `--use_lstm` for the LSTM
* `--use_delta` for the Delta Net
* `--use_lt` for the Linear Transformer (LT)
* `--use_rec_delta` for the Recurrent Delta Net (RDN)
* `--use_delta_rnn` for the Delta RNN

To specify the environment/game to be used,
look up the name in `list_games.txt` and give it to the `--env` flag, and the corresponding
number of actions to `--num_actions`.

Depending on the machine, we experienced that appending `CUDA_LAUNCH_BLOCKING=1` was necessary
to run the training script above.

## Evaluation

For evaluation, directly call `polybeast_learner` with `--mode test` flag.
```
export TORCH_EXTENSIONS_DIR="/home/me/torch_extensions/torchbeast"
export CUDA_VISIBLE_DEVICES=2

SAVE_DIR=saved_models

python -m torchbeast.polybeast_learner \
     --mode test \
     --env GopherNoFrameskip-v4 \
     --num_actors 48 \
     --num_servers 48 \
     --total_steps 200000000 \
     --save_extra_checkpoint 50000000 \
     --learning_rate 0.0006 \
     --grad_norm_clipping 40 \
     --epsilon 0.01 \
     --entropy_cost 0.01 \
     --batch_size 32 \
     --unroll_length 50 \
     --num_actions 8 \
     --use_rec_delta \
     --num_layers 2 \
     --num_learner_threads 1 \
     --num_inference_threads 1 \
     --xpid "gopher_rec_delta" \
     --savedir ${SAVE_DIR}
```

### Evaluation by excluding negative scores for some environments
For *Battlezone* and *Up'n Down*, we found that well trained models sometimes result in a very negative reward, which seems to be a bug of the environment
(see https://github.com/openai/gym/issues/2233).
Therefore for these two environments, we run evaluation by excluding the episodes giving negative scores (i.e. run the evaluation until we get 30 episodes with a positive final score) by calling `torchbeast.noneg_polybeast_learner` instead of `torchbeast.polybeast_learner` in the script above.
