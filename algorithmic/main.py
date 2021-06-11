# Main code to be executed
import os
import time
from datetime import datetime
import argparse
import logging
import random
import subprocess

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import LTEDataset
from model import (
    LSTMModel, TrafoModel, LinearTransformer, DeltaNetModel,
    OwnTransformer, FastRNNModel, FastFFslowRNNModel, RecDeltaNetModel)
from eval_utils import compute_accuracy


DEVICE = 'cuda'

model_name_map = ['lstm',  # 0
                  'trafo',  # 1
                  'delta-net',  # 2
                  'fast-rnn',  # 3
                  'fast-lstm',  # 4
                  'fast-ff-slow-rnn',  # 5
                  'rec-update',  # 6
                  'rec-delta',  # 7
                  'linear-trafo',  # 8
                  'own-trafo',  # 9
                  ]

parser = argparse.ArgumentParser(description='Learning to execute')
parser.add_argument('--data_dir', type=str,
                    default='/home/me/src/learn-to-execute/utils/data_v3/',
                    help='location of the data corpus')
parser.add_argument('--level', type=int, default=3,
                    choices=[3, 5, 10],
                    help='Number of variables (3, 5 or 10)')
parser.add_argument('--work_dir', default='save_models', type=str,
                    help='where to save model ckpt.')
parser.add_argument('--model_type', type=int, default=0,
                    choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    help='0: LSTM, 1: Trafo, 2: Linear-Trafo, '
                    '3: fast RNN, 4: fast LSTM, 5: fast FF-slow RNN...'
                    '6: recurrent update, 7: 6 w. tanh')
parser.add_argument('--seed', default=1, type=int, help='Seed.')
parser.add_argument('--debug', action='store_true', 
                    help='Print training loss after each batch.')
# model hyper-parameters:
parser.add_argument('--num_layer', default=2, type=int,
                    help='number of layers. for both LSTM and Trafo.')
parser.add_argument('--hidden_size', default=512, type=int,
                    help='hidden size. for both LSTM and Trafo.')
parser.add_argument('--emb_size', default=128, type=int,
                    help='emb size. for LSTM.')
parser.add_argument('--n_head', default=8, type=int,
                    help='Transformer number of heads.')
parser.add_argument('--ff_factor', default=4, type=int,
                    help='Transformer ff dim to hidden dim ratio.')
parser.add_argument('--remove_pos_enc', action='store_true',
                    help='Remove postional encoding in Trafo.')
parser.add_argument('--dropout', default=0.0, type=float,
                    help='dropout rate.')
# training hyper-parameters:
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size.')
parser.add_argument('--learning_rate', default=1e-4, type=float,
                    help='batch size.')
parser.add_argument('--grad_cummulate', default=1, type=int,
                    help='number of gradient accumulation steps.')
parser.add_argument('--num_epoch', default=200, type=int,
                    help='number of training epochs.')
parser.add_argument('--report_every', default=200, type=int,
                    help='Report valid acc every this steps (not used).')
parser.add_argument('--clip', default=0.0, type=float,
                    help='global norm clipping threshold.')
# for wandb
parser.add_argument('--project_name', type=str, default=None,
                    help='project name for wandb.')
parser.add_argument('--job_name', type=str, default=None,
                    help='job name for wandb.')
parser.add_argument('--use_wandb', action='store_true',
                    help='use wandb.')

args = parser.parse_args()

# Set seed
torch.manual_seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Set work directory
args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)

# logging
log_file_name = f"{args.work_dir}/log.txt"
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO, format='%(message)s', handlers=handlers)

loginf = logging.info

loginf(f"torch version: {torch.__version__}")
# loginf(f"Last commit: {subprocess.check_output(['git', 'rev-parse', 'HEAD'])}")
loginf(f"Work dir: {args.work_dir}")

model_name = model_name_map[args.model_type]

# wandb settings
if args.use_wandb:  # configure wandb.
    import wandb
    use_wandb = True

    if args.project_name is None:
        project_name = (os.uname()[1]
                        + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        project_name = args.project_name

    wandb.init(project=project_name)

    if args.job_name is None:
        wandb.run.name = f"{os.uname()[1]}//{model_name}-{args.model_type}//" \
                         f"level{args.level}//seed{args.seed}/" \
                         f"L{args.num_layer}/h{args.hidden_size}/" \
                         f"e{args.emb_size}/" \
                         f"n{args.n_head}/ff{args.ff_factor}/" \
                         f"d{args.dropout}/b{args.batch_size}/" \
                         f"lr{args.learning_rate}/pos{args.remove_pos_enc}/" \
                         f"g{args.grad_cummulate}/ep{args.num_epoch}"
    else:
        wandb.run.name = f"{os.uname()[1]}//{args.job_name}"

    config = wandb.config
    config.host = os.uname()[1]  # host node name
    config.data_dir = args.data_dir
    config.seed = args.seed
    config.level = args.level
    config.work_dir = args.work_dir
    config.model_type = args.model_type
    config.hidden_size = args.hidden_size
    config.emb_size = args.emb_size
    config.n_head = args.n_head
    config.ff_factor = args.ff_factor
    config.dropout = args.dropout
    config.remove_pos_enc = args.remove_pos_enc
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.grad_cummulate = args.grad_cummulate
    config.num_epoch = args.num_epoch
    config.report_every = args.report_every
else:
    use_wandb = False


# Set paths
data_path = args.data_dir

src_pad_idx = 0  # src_pad does not matter; as padding is aligned.
tgt_pad_idx = -1  # to be passed to the loss func.

# train_batch_size = 64
train_batch_size = args.batch_size
valid_batch_size = train_batch_size
test_batch_size = train_batch_size

train_file_src = f"{data_path}/train_{args.level}.src"
train_file_tgt = f"{data_path}/train_{args.level}.tgt"

valid_file_src = f"{data_path}/valid_{args.level}.src"
valid_file_tgt = f"{data_path}/valid_{args.level}.tgt"

test_file_src = f"{data_path}/test_{args.level}.src"
test_file_tgt = f"{data_path}/test_{args.level}.tgt"

# Construct dataset
train_data = LTEDataset(src_file=train_file_src, tgt_file=train_file_tgt,
                        src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx,
                        src_vocab=None, tgt_vocab=None)

src_vocab = train_data.src_vocab
tgt_vocab = train_data.tgt_vocab

no_print_idx = tgt_vocab.get_no_op_id()  # Used to compute print accuracy.

valid_data = LTEDataset(src_file=valid_file_src, tgt_file=valid_file_tgt,
                        src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx,
                        src_vocab=src_vocab, tgt_vocab=tgt_vocab)

# Set dataloader
train_data_loader = DataLoader(
    dataset=train_data, batch_size=train_batch_size, shuffle=True)
valid_data_loader = DataLoader(
    dataset=valid_data, batch_size=valid_batch_size, shuffle=False)

model_type = args.model_type  # 0 for LSTM, 1 for regular Trafo

if model_type == 0:
    # LSTM params:
    emb_dim = args.emb_size
    hidden_size = args.hidden_size
    num_layers = args.num_layer
    dropout = args.dropout
elif model_type == 1:
    # Trafo params:
    hidden_size = args.hidden_size
    dropout = args.dropout
    nheads = args.n_head
    num_layers = args.num_layer
    ff_factor = args.ff_factor
    use_pos_enc = not args.remove_pos_enc
elif model_type in [2, 3, 4, 5, 6, 7, 8, 9]:
    # Trafo params:
    hidden_size = args.hidden_size
    dropout = args.dropout
    nheads = args.n_head
    num_layers = args.num_layer
    ff_factor = args.ff_factor
    use_pos_enc = not args.remove_pos_enc

# Common params:
in_vocab_size = src_vocab.size()
out_vocab_size = tgt_vocab.size()

loginf(f"Input vocab size: {in_vocab_size}")
loginf(f"Output vocab size: {out_vocab_size}")

# model
if model_type == 0:
    loginf("Model: LSTM")
    model = LSTMModel(emb_dim=emb_dim, hidden_size=hidden_size,
                      num_layers=num_layers, in_vocab_size=in_vocab_size,
                      out_vocab_size=out_vocab_size, dropout=dropout)

elif model_type == 1:
    loginf("Model: Transformer")
    model = TrafoModel(hidden_size=hidden_size, in_vocab_size=in_vocab_size,
                       out_vocab_size=out_vocab_size, dropout=dropout,
                       nheads=nheads, num_layers=num_layers,
                       ff_factor=ff_factor)

elif model_type in [2, 3, 4, 5, 6, 7, 8, 9]:  # Fast weight models.
    # We initially had more models, but we did not get a chance to try all of
    # them. For the release, we only kept those used in the paper and removed
    # others to save kernel compilation time.
    if model_type == 2:
        loginf("Model: Delta Net")
        model_func = DeltaNetModel
    elif model_type == 3:
        loginf("Model: Fast RNN i.e. Delta RNN")
        model_func = FastRNNModel
    elif model_type == 4:
        # this was delta LSTM, but was not used in the end.
        assert False, "Removed."
    elif model_type == 5:
        loginf("Model: Fast-FF/Slow-RNN")
        model_func = FastFFslowRNNModel
    elif model_type == 6:
        assert False, "Removed."  # was like below but without tanh
    elif model_type == 7:
        loginf("Model: Recurrent Delta Net")
        model_func = RecDeltaNetModel
    elif model_type == 8:
        loginf("Model: Linear Transformer")
        model_func = LinearTransformer
    elif model_type == 9:
        loginf("Model: Own Transformer")
        model_func = OwnTransformer
    else:
        assert False, "Just in case."

    assert hidden_size % nheads == 0
    dim_head = int(hidden_size / nheads)

    model = model_func(in_vocab_size=in_vocab_size,
                       out_vocab_size=out_vocab_size,
                       hidden_size=hidden_size, num_layers=num_layers,
                       num_head=nheads, dim_head=dim_head,
                       dim_ff=ff_factor * hidden_size,
                       dropout=dropout, use_pos_enc=use_pos_enc)
else:
    assert False, f"Unexpected model_type: {model_type}"

loginf(f"Number of trainable params: {model.num_params()}")
loginf(f"{model}")

model = model.to(DEVICE)


# Optimization settings:
num_epoch = args.num_epoch
grad_cummulate = args.grad_cummulate
loginf(f"Batch size: {train_batch_size}")
loginf(f"Gradient accumulation for {grad_cummulate} steps.")
loginf(f"Seed: {args.seed}")
learning_rate = args.learning_rate

loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate,
                             betas=(0.9, 0.995), eps=1e-9)
clip = args.clip

loginf(f"Learning rate: {learning_rate}")
loginf(f"clip at: {clip}")

# Training
acc_loss = 0.
steps = 0
stop_acc = 0.8
best_val_acc = 0.0
best_epoch = 1
check_between_epochs = False
report_every = args.report_every

best_model_path = os.path.join(args.work_dir, 'best_model.pt')
lastest_model_path = os.path.join(args.work_dir, 'lastest_model.pt')

loginf(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] Start training")
start_time = time.time()
interval_start_time = time.time()

# Re-seed so that the order of data presentation
# is determined by the seed independent of the model choice.
torch.manual_seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

for ep in range(num_epoch):
    for idx, batch in enumerate(train_data_loader):
        model.train()

        src, tgt = batch
        logits = model(src)
        logits = logits.contiguous().view(-1, logits.shape[-1])
        labels = tgt.view(-1)
        loss = loss_fn(logits, labels)

        loss.backward()

        if idx % grad_cummulate == 0:
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            model.reset_grad()

        with torch.no_grad():
            acc_loss += loss
            steps += 1
            if args.debug:
                loginf(src)
                loginf(f"loss: {loss}")

        if check_between_epochs:
            assert False, "Not implemented"
            loginf(f"[ep {ep+1} step {idx}] "
                   f"train loss: {acc_loss/steps} -----")
            v_loss, v_acc, v_acc_noop, v_acc_print = compute_accuracy(
                model, valid_data_loader, loss_fn, no_print_idx=no_print_idx,
                pad_value=tgt_pad_idx, show_example=False)

            loginf(f"[ep {ep+1} step {idx}] val loss: {v_loss}")
            loginf(f"[ep {ep+1} step {idx}] val ac: {v_acc:.2f} %")
            loginf(f"[ep {ep+1} step {idx}] val no-op ac: {v_acc_noop:.2f} %")
            loginf(f"[ep {ep+1} step {idx}] val print ac: {v_acc_print:.2f} %")

    with torch.no_grad():
        loginf(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
               f"End epoch {ep+1} =============")
        loginf(f"train loss: {acc_loss / steps}")

        v_loss, v_acc, v_acc_noop, v_acc_print = compute_accuracy(
            model, valid_data_loader, loss_fn, no_print_idx=no_print_idx,
            pad_value=tgt_pad_idx, show_example=False)
        loginf(f"valid loss: {v_loss}")
        loginf(f"valid acc: {v_acc:.2f} %")
        loginf(f"valid no-op acc: {v_acc_noop:.2f} %")
        loginf(f"valid print acc: {v_acc_print:.2f} %")

        if use_wandb:
            wandb.log({"train_loss": acc_loss / steps})
            wandb.log({"valid_loss": v_loss})
            wandb.log({"valid_acc": v_acc})
            wandb.log({"valid_acc_noop": v_acc_noop})
            wandb.log({"valid_acc_print": v_acc_print})

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch = ep + 1
            # Save the best model
            loginf("The best model so far.")
            torch.save({'epoch': best_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'valid_acc': best_val_acc}, best_model_path)
            loginf("Saved.")
        # Save the latest model
        torch.save({'epoch': ep + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_acc': v_acc}, lastest_model_path)

        acc_loss = 0.0
        steps = 0


elapsed = time.time() - start_time
loginf(f"Ran {num_epoch} epochs in {elapsed / 60.:.2f} min.")
loginf(f"Best validation acc: {best_val_acc:.2f}")

if best_epoch > 1:  # load the best model and evaluate on the test set
    del train_data_loader, train_data

    test_data = LTEDataset(src_file=test_file_src, tgt_file=test_file_tgt,
                           src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx,
                           src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    test_data_loader = DataLoader(
        dataset=test_data, batch_size=test_batch_size, shuffle=False)

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        test_loss, test_acc, test_acc_char, test_acc_print = compute_accuracy(
            model, test_data_loader, loss_fn, no_print_idx=no_print_idx,
            pad_value=tgt_pad_idx, show_example=False)

    loginf(f"Final model test acc: {test_acc:.2f} %")
