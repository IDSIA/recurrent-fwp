# Initial code from https://github.com/kimiyoung/transformer-xl
# Changes related to fast weights and support for wandb added by Kazuki Irie

import argparse
import time
import math
import os, sys
import subprocess
import itertools
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from model_main import MemTransformerLM

from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel


parser = argparse.ArgumentParser(
    description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=50,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=500,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=1000,
                    help='inner dimension in FF')
parser.add_argument('--d_res', type=int, default=None,
                    help='res connection size')
parser.add_argument('--remove_ff', action='store_true',
                    help='do not use feed-forward layers')
parser.add_argument('--remove_lnorm', action='store_true',
                    help='do not use lnorm (only for RNNs for now)')
parser.add_argument('--remove_res', action='store_true',
                    help='do not use res connection (only for RNNs for now)')
parser.add_argument('--remove_out_proj', action='store_true',
                    help='remove proj after RNN layer (only for RNNs)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)'
                    )
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)'
                    )
parser.add_argument('--init_std', type=float, default=0.02,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim', default='adam', type=str,
                    choices=['adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=100000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10,
                    help='evaluation batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')
parser.add_argument('--tgt_len', type=int, default=70,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=50,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--div_val', type=int, default=1,
                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=4000,
                    help='evaluation interval')
parser.add_argument('--work_dir', default='LM-TFM', type=str,
                    help='experiment directory.')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_model', type=str, default=None,
                    help='full path to the model checkpoint file')
parser.add_argument('--restart_opt', type=str, default=None,
                    help='full path to the training state checkpoint file')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                    '2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--sample_softmax', type=int, default=-1,
                    help='number of samples in sampled softmax')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--finetune_v2', action='store_true',
                    help='finetune v2')
parser.add_argument('--finetune_v3', action='store_true',
                    help='finetune v3')
parser.add_argument('--performer_proj_dim', type=int, default=16,
                    help='projection dimension for performer layers.')
parser.add_argument('--dpfp_n_roll', type=int, default=2,
                    help='number of rolls for dpfp attention layers.')
parser.add_argument('--fast_net_depth', type=int, default=1,
                    help='number of layers in the fast nets.')
parser.add_argument('--use_slow_base_weights', action='store_true',
                    help='use base slow weights in fast net.')                  
parser.add_argument('--carry_over_fast_weight', action='store_true',
                    help='carry over fast weights.')
parser.add_argument('--skip_attn_normalization', action='store_true',
                    help='skip denominator in fast weights.')
parser.add_argument('--no_pos', action='store_true',
                    help='do not use positional encoding.')
parser.add_argument('--fp16', action='store_true',
                    help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can '
                    'improve fp16 convergence.')
parser.add_argument(
    '--dynamic-loss-scale', action='store_true',
    help='Use dynamic loss scaling.'
         'If supplied, this argument supersedes --static-loss-scale.')
parser.add_argument('--project_name', type=str, default=None,
                    help='project name for wandb.')
parser.add_argument('--job_name', type=str, default=None,
                    help='job name for wandb.')
parser.add_argument('--use_wandb', action='store_true',
                    help='use wandb.')

args = parser.parse_args()

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
        # wandb.run.name = (os.uname()[1]
        #                   + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #                   + args.work_dir)
        wandb.run.name = f"{os.uname()[1]}//{args.attn_type}//" \
                         f"{args.performer_proj_dim}//" \
                         f"{args.tgt_len}-{args.eval_tgt_len}" \
                         f"-{args.mem_len}//" \
                         f"{args.n_layer}-{args.n_head}-{args.d_res}" \
                         f"{args.d_head}-{args.d_embed}-{args.d_model}-" \
                         f"{args.d_inner}-{args.dropout}-{args.dropatt}-//" \
                         f"{args.lr}-{args.warmup_step}" \
                         f"{args.batch_size}-{args.eval_batch_size}//" \
                         f"{args.seed}-{args.work_dir}-{args.dpfp_n_roll}" \
                         f"-{args.carry_over_fast_weight}-{args.no_pos}" \
                         f"-{args.fast_net_depth}-{args.use_slow_base_weights}"
    else:
        wandb.run.name = f"{os.uname()[1]}//{args.job_name}"

    config = wandb.config
    config.host = os.uname()[1]  # host node name
    config.data=args.data
    config.dataset=args.dataset
    config.n_layer=args.n_layer
    config.n_head=args.n_head
    config.d_head=args.d_head
    config.d_embed=args.d_embed
    config.d_model=args.d_model
    config.d_inner=args.d_inner
    config.d_res = args.d_res
    config.dropout=args.dropout
    config.dropatt=args.dropatt
    config.init=args.init
    config.emb_init=args.emb_init
    config.init_range=args.init_range
    config.emb_init_range=args.emb_init_range
    config.init_std=args.init_std
    config.proj_init_std=args.proj_init_std
    config.optim=args.optim
    config.lr=args.lr
    config.mom=args.mom
    config.scheduler=args.scheduler
    config.warmup_step=args.warmup_step
    config.decay_rate=args.decay_rate
    config.lr_min=args.lr_min
    config.clip=args.clip
    config.clip_nonemb=args.clip_nonemb
    config.max_step=args.max_step
    config.batch_size=args.batch_size
    config.eval_batch_size=args.eval_batch_size
    config.batch_chunk=args.batch_chunk
    config.tgt_len=args.tgt_len
    config.eval_tgt_len=args.eval_tgt_len
    config.ext_len=args.ext_len
    config.mem_len=args.mem_len
    config.not_tied=args.not_tied
    config.seed=args.seed
    config.cuda=args.cuda
    config.adaptive=args.adaptive
    config.div_val=args.div_val
    config.pre_lnorm=args.pre_lnorm
    config.varlen=args.varlen
    config.multi_gpu=args.multi_gpu
    config.log_interval=args.log_interval
    config.eval_interval=args.eval_interval
    config.work_dir=args.work_dir
    config.restart=args.restart
    config.restart_model = args.restart_model
    config.restart_opt = args.restart_opt
    config.debug=args.debug
    config.same_length=args.same_length
    config.attn_type=args.attn_type
    config.clamp_len=args.clamp_len
    config.eta_min=args.eta_min
    config.gpu0_bsz=args.gpu0_bsz
    config.max_eval_steps=args.max_eval_steps
    config.sample_softmax=args.sample_softmax
    config.patience=args.patience
    config.finetune_v2=args.finetune_v2
    config.finetune_v3=args.finetune_v3
    config.performer_proj_dim=args.performer_proj_dim
    config.dpfp_n_roll=args.dpfp_n_roll
    config.use_slow_base_weights=args.use_slow_base_weights
    config.carry_over_fast_weight=args.carry_over_fast_weight
    config.skip_attn_normalization=args.skip_attn_normalization
    config.remove_ff = args.remove_ff
    config.remove_res = args.remove_res
    config.remove_lnorm = args.remove_lnorm
    config.remove_out_proj = args.remove_out_proj
    config.no_pos=args.no_pos
    config.fast_net_depth=args.fast_net_depth
    config.fp16=args.fp16
    config.static_loss_scale=args.static_loss_scale
else:
    use_wandb = False

args.tied = not args.not_tied

if args.d_embed < 0:
    args.d_embed = args.d_model

if args.d_res is not None and args.attn_type in [81, 91]:
    assert args.d_res == args.d_model, "`d_res = d_model` required for 81 & 91"

assert args.ext_len >= 0, 'extended context length must be non-negative'
assert args.batch_size % args.batch_chunk == 0

if args.carry_over_fast_weight:
    assert args.mem_len == 0, "Incompatible"

args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
logging = create_exp_dir(args.work_dir, scripts_to_save=None, debug=args.debug)

logging(f'torch version: {torch.__version__}')
logging(
    f"Last commit: {subprocess.check_output(['git', 'rev-parse', 'HEAD'])}")

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, '
              'so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

# Validate `--fp16` option
if args.fp16:
    if not args.cuda:
        print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
        args.fp16 = False
    else:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

device = torch.device('cuda' if args.cuda else 'cpu')

###############################################################################
# Load data
###############################################################################
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_token = ntokens

tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)

va_iter = corpus.get_iterator('valid', args.eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)

te_iter = corpus.get_iterator('test', args.eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]

if args.adaptive:
    assert args.dataset in ['wt103', 'lm1b']

    if args.dataset == 'wt103':
        cutoffs = [20000, 40000, 200000]
        tie_projs += [True] * len(cutoffs)

    elif args.dataset == 'lm1b':
        cutoffs = [60000, 100000, 640000]
        tie_projs += [False] * len(cutoffs)


###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)

    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)

    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)

    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)

    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)

    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout


def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt


model = MemTransformerLM(
    ntokens,
    args.n_layer,
    args.n_head,
    args.d_model,
    args.d_head,
    args.d_inner,
    args.dropout,
    args.dropatt,
    tie_weight=args.tied,
    d_embed=args.d_embed,
    div_val=args.div_val,
    tie_projs=tie_projs,
    pre_lnorm=args.pre_lnorm,
    remove_ff=args.remove_ff,
    remove_res=args.remove_res,
    remove_lnorm=args.remove_lnorm,
    remove_out_proj=args.remove_out_proj,
    d_res=args.d_res,
    tgt_len=args.tgt_len,
    ext_len=args.ext_len,
    mem_len=args.mem_len,
    cutoffs=cutoffs,
    same_length=args.same_length,
    attn_type=args.attn_type,
    clamp_len=args.clamp_len,
    sample_softmax=args.sample_softmax,
    proj_dim=args.performer_proj_dim,
    n_roll=args.dpfp_n_roll,
    skip_attn_normalization=args.skip_attn_normalization,
    no_pos=args.no_pos,
    fast_net_depth=args.fast_net_depth,
    use_slow_base_weights=args.use_slow_base_weights,
    device=device,
)
logging('=' * 100)
logging(f"{print(model.word_emb)}")
logging(f"{print(model.layers)}")
if model.crit is not None:
    logging(f"{print(model.crit.out_layers)}")


if args.restart:
    assert args.restart_model is not None, "restart_model required to restart"
    assert args.restart_opt is not None, "restart_opt required to restart"
    ckpt_path = args.restart_model
    opt_path = args.restart_opt

    logging(f"[Restart] Load model from: {ckpt_path}")
    opt_checkpoint = torch.load(opt_path)
    last_ep = opt_checkpoint['epoch']
    last_val_ppl = opt_checkpoint['val_ppl']
    b_val_ppl = opt_checkpoint['best_val_ppl']
    b_ep = opt_checkpoint['best_epoch']
    logging(f"[Restart] Last epoch: {last_ep}, valid ppl: {last_val_ppl:.2f}, "
            f"best val ppl so far: {b_val_ppl:.2f} at epoch {b_ep}")
    with open(ckpt_path, 'rb') as f:
        model = torch.load(f)
    # model.load_state_dict(checkpoint['model_state_dict'])

    if not args.fp16:
        model = model.float()

    model.apply(update_dropout)
    model.apply(update_dropatt)
else:
    model.apply(weights_init)

    # ensure embedding init is not overridden by
    # out_layer in case of weight sharing
    model.word_emb.apply(weights_init)

args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

if args.fp16:
    model = model.half()

if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0:
        para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                          model, dim=1).to(device)
    else:
        para_model = nn.DataParallel(model, dim=1).to(device)
else:
    para_model = model.to(device)


# optimizer
if args.optim.lower() == 'sgd':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
        optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
            momentum=args.mom)

elif args.optim.lower() == 'adam':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
        optimizer = optim.Adam(dense_params, lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

elif args.optim.lower() == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

# scheduler
if args.scheduler == 'cosine':
    # here we do not set eta_min to lr_min to be backward compatible
    # because in previous versions eta_min is default to 0
    # rather than the default value of lr_min 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
        args.max_step, eta_min=args.eta_min)  # should use eta_min arg
    if args.sample_softmax > 0:
        # should use eta_min arg
        scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_sparse, args.max_step, eta_min=args.eta_min)

elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step \
                   else step / (args.warmup_step ** 1.5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)

    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_sparse, factor=args.decay_rate,
            patience=args.patience, min_lr=args.lr_min)

elif args.scheduler == 'constant':
    scheduler = None
    pass

if args.cuda and args.fp16:
    # If args.dynamic_loss_scale is False, static_loss_scale will be used.
    # If args.dynamic_loss_scale is True,
    # it will take precedence over static_loss_scale.
    optimizer = FP16_Optimizer(optimizer,
                               static_loss_scale=args.static_loss_scale,
                               dynamic_loss_scale=args.dynamic_loss_scale,
                               dynamic_loss_args={'init_scale': 2 ** 16})

if args.restart:
    optimizer.load_state_dict(opt_checkpoint['optimizer_state_dict'])
    logging(f"[Restart] Load optimizer states from: {opt_path}")
    args.warmup_step = 0
    logging("[Restart] Set warmup step to 0 for warm restarting.")
    if scheduler is not None:
        scheduler.load_state_dict(opt_checkpoint['scheduler_state_dict'])
        logging(f"[Restart] Load scheduler states from: {opt_path}")

logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))

logging('=' * 100)
logging('#params = {}'.format(args.n_all_param))
logging('#non emb params = {}'.format(args.n_nonemb_param))

###############################################################################
# Training code
###############################################################################


def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.reset_length(
            args.eval_tgt_len,  # tgt_len
            args.ext_len + args.tgt_len - args.eval_tgt_len,
            args.mem_len)  # mem_len
    else:
        model.reset_length(args.eval_tgt_len,
                           args.ext_len,
                           args.mem_len + args.tgt_len - args.eval_tgt_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = tuple()
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            ret = model(data, target, *mems,
                        carry_over_fast_weight=args.carry_over_fast_weight)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_loss / total_len


def train():
    global train_step, train_loss, best_val_loss, eval_start_time, best_epoch
    global log_start_time

    model.train()  # Turn on training mode which enables dropout.

    if args.batch_chunk > 1:
        mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        mems = tuple()

    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter

    for batch, (data, target, seq_len) in enumerate(train_iter):
        model.zero_grad()

        if args.batch_chunk > 1:
            data_chunks = torch.chunk(data, args.batch_chunk, 1)
            target_chunks = torch.chunk(target, args.batch_chunk, 1)

            for i in range(args.batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                ret = para_model(
                    data_i, target_i, *mems[i],
                    carry_over_fast_weight=args.carry_over_fast_weight)
                loss, mems[i] = ret[0], ret[1:]
                loss = loss.float().mean().type_as(loss) / args.batch_chunk
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                train_loss += loss.float().item()
        else:
            ret = para_model(
                data, target, *mems,
                carry_over_fast_weight=args.carry_over_fast_weight)
            loss, mems = ret[0], ret[1:]
            loss = loss.float().mean().type_as(loss)
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            train_loss += loss.float().item()

        if args.fp16:
            optimizer.clip_master_grads(args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if args.sample_softmax > 0:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)

        if train_step % args.log_interval == 0:
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches |' \
                      ' lr {:.3g} | ms/batch {:5.2f} | loss {:5.2f}'.format(
                epoch, train_step, batch+1, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
                if use_wandb:
                    wandb.log({"bpc": cur_loss / math.log(2)})
            else:
                log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
                if use_wandb:
                    wandb.log({"ppl": math.exp(cur_loss)})
                assert not math.isnan(math.exp(cur_loss))
            logging(log_str)
            train_loss = 0
            log_start_time = time.time()

        if train_step % args.eval_interval == 0:
            val_loss = evaluate(va_iter)
            logging('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.2f}'.format(
                train_step // args.eval_interval, train_step,
                (time.time() - eval_start_time), val_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
                if use_wandb:
                    wandb.log({"valid_bpc": val_loss / math.log(2)})
            else:
                val_ppl = math.exp(val_loss)
                log_str += ' | valid ppl {:9.3f}'.format(val_ppl)
                if use_wandb:
                    wandb.log({"valid_ppl": val_ppl})
                assert not math.isnan(val_ppl)

            logging(log_str)
            logging('-' * 100)

            # dev-performance based learning rate annealing
            if args.scheduler == 'dev_perf':
                scheduler.step(val_loss)
                if args.sample_softmax > 0:
                    scheduler_sparse.step(val_loss)

            # Save the model if the validation loss is the best so far.
            if not best_val_loss or val_loss < best_val_loss:
                if not args.debug:
                    best_epoch = epoch
                    logging(f"Best val achieved at epoch {epoch} "
                            f'{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
                    # Save full model: structure and params.
                    ckpt_path = os.path.join(args.work_dir, 'best_model.pt')
                    logging(f"Saving ckpt to {ckpt_path}")
                    with open(ckpt_path, 'wb') as f:
                        torch.save(model, f)
                    # Save training states: optimizer and scheduler states
                    # for eventual restart.
                    opt_path = os.path.join(args.work_dir, 'best_opt.pt')
                    logging(f"Saving training states to {opt_path}")
                    torch.save({'epoch': epoch,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'val_ppl': val_ppl}, opt_path)
                best_val_loss = val_loss

            ckpt_path = os.path.join(args.work_dir, 'latest_model.pt')
            logging(f"Saving the latest ckpt to {ckpt_path}")
            with open(ckpt_path, 'wb') as f:
                torch.save(model, f)
            opt_path = os.path.join(args.work_dir, 'latest_opt.pt')
            logging(f"Saving the latest training states to {opt_path}")
            torch.save({'epoch': epoch,
                        'best_epoch': best_epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_ppl': math.exp(best_val_loss),
                        'val_ppl': val_ppl}, opt_path)
            logging('-' * 100)

            eval_start_time = time.time()

        if train_step == args.max_step:
            break


# Loop over epochs.
train_step = 0
train_loss = 0
best_val_loss = None
best_epoch = 0

log_start_time = time.time()
eval_start_time = time.time()

logging(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in itertools.count(start=1):
        train()
        logging(f'end of epoch {epoch}: '
                f'{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
        if train_step == args.max_step:
            logging('-' * 100)
            logging('End of training')
            break

except KeyboardInterrupt:
    logging('-' * 100)
    logging('Exiting from training early')

# Load the best saved model.
logging('Evaluation...')
ckpt_path = os.path.join(args.work_dir, 'best_model.pt')
logging(f'Load the best ckpt from: {ckpt_path}')
opt_path = os.path.join(args.work_dir, 'best_opt.pt')
opt_checkpoint = torch.load(opt_path)
best_val_ppl = opt_checkpoint['val_ppl']
best_ep = opt_checkpoint['epoch']
logging(f'The best valid ppl: {best_val_ppl:.2f} at epoch: {best_ep}')

with open(ckpt_path, 'rb') as f:
    model = torch.load(f)

model = model.to(device)

# Run on test data.
logging('Evaluation...')
test_loss = evaluate(te_iter)
logging('=' * 100)
if args.dataset in ['enwik8', 'text8']:
    logging('| End of training | test loss {:5.2f} | test bpc {:9.5f}'.format(
        test_loss, test_loss / math.log(2)))
else:
    logging('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(
        test_loss, math.exp(test_loss)))

logging('=' * 100)
