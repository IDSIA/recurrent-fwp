# Code Execution & Sequential ListOps

This directory contains code we used for the two algorithmic tasks: code execution and sequential ListOps.

See Appendix B in our paper for the task descriptions and examples.

## Data Generation
* **Code execution**:
```
cd utils
# Set `max_num_vars` in data_generator.py to either 3 or 5
python data_generator.py  --dump_dir my_data_dir --code_length 100  # `code_length` is the number of statements
```
The dataset can be further customized by changing `max_num_vars` or `--code_length` i.e. the number of statements per sequence.
It should also not be difficult to extend it to support other statement types.

* **ListOps**:
```
cd utils
python nyu_listops.py --dump_dir my_data_dir --only_depth 11  # data for depth 10
python nyu_listops.py --dump_dir my_data_dir --only_depth 16  # data for depth 15
```
For further options to specify the properties of the ListOps dataset (such as the maximum number of arguments `MAX_ARGS`,
maximum or minimum length `MAX_LENGTH`/`MIN_LENGTH` or choice of list operations `OPERATORS` etc) see `nyu_listops.py`.

## Requirements
* PyTorch (PyT >= 1.6.0 recommended)
* Ninja to compile custom CUDA kernels (`pip install ninja`)
* Optionally: wandb for monitoring jobs (or disable it by removing the `--use_wandb` flag; see below)


## Training
A generic script to train Transformer model variants on the code execution task is as follows.
For ListOps, replace `main.py` by `listops_main.py` and
provide the data file prefix to the `level` argument.

Separate paths for `TORCH_EXTENSIONS_DIR` should be used for code execution and ListOps.

`model_type` specifies the model type. 
The models used in the paper are as follows:
* `0`: LSTM
* `1`: Regular Transformer
* `2`: Delta Net
* `3`: Delta RNN
* `7`: Recurrent Delta Net
* `8`: Linear Transformer

```
export TORCH_EXTENSIONS_DIR="my_dir/torch_extensions2"
DATA_DIR='my_data_dir'

python main.py \
  --data_dir ${DATA_DIR} \
  --level 3 \
  --model_type 2 \
  --num_layer 4 \
  --hidden_size 256 \
  --n_head 16 \
  --ff_factor 4 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 3e-4 \
  --seed 11 \
  --grad_cummulate 1 \
  --num_epoch 200 \
  --project_name "my_project" \
  --use_wandb \
  --remove_pos_enc
```

## Evaluation
Evalution is automatically run at the end of training using the best performing checkpoint based on the validation accuracy.

## References
* https://github.com/wojciechz/learning_to_execute
* Code execution task is designed based on the description by Fan et al. (no public code available)
