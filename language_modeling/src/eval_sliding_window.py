# Initial code from https://github.com/kimiyoung/transformer-xl
# `evaluate` function modified by Kazuki Irie
# 
# Evaluation of perplexity using sliding window of contexts.
#
# Except for the first segment(s) of the text, perplexity computations at all
# positions have access to the same context size specified as the window size.
# i.e. for each window position, only the last position is used in the ppl
# computation. This is a very slow evaluation for standard transformers
# but crucial to avoid "short context" at the segment beginning in the
# standard evaluation.

import argparse
import time
import math
import os

from tqdm import tqdm
import torch

from data_utils import get_lm_corpus
from utils.exp_utils import get_logger

parser = argparse.ArgumentParser(
    description='PyTorch Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
parser.add_argument('--split', type=str, default='all',
                    choices=['all', 'valid', 'test'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=5,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--work_dir', type=str, required=True,
                    help='path to the work_dir')
parser.add_argument('--model_file', type=str, required=False, default=None,
                    help='full path of the model file')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--skip_first', action='store_true',
                    help='old behavior, skip tgt_len first tokens')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')

args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'

device = torch.device("cuda" if args.cuda else "cpu")

# Get logger
logging = get_logger(os.path.join(args.work_dir, 'log.txt'),
                     log_=not args.no_log)

# Load dataset
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)

# Load the best saved model.
if args.model_file is not None:
    ckpt_path = args.model_file
else:  # assume `work_dir` to contain `best_model.pt`
    ckpt_path = os.path.join(args.work_dir, 'best_model.pt')
    opt_path = os.path.join(args.work_dir, 'best_opt.pt')
    assert os.path.exists(opt_path)
    opt_checkpoint = torch.load(opt_path)
    best_val_ppl = opt_checkpoint['val_ppl']
    best_ep = opt_checkpoint['epoch']
    logging(f'The best valid ppl: {best_val_ppl:.2f} at epoch: {best_ep}')

assert os.path.exists(ckpt_path)
print(f'Loading checkpoint from: {ckpt_path}')
with open(ckpt_path, 'rb') as f:
    model = torch.load(f)

model.backward_compatible()
model = model.to(device)

logging(f'Evaluating with bsz {args.batch_size} tgt_len {args.tgt_len} '
        f'ext_len {args.ext_len} mem_len {args.mem_len} '
        f'clamp_len {args.clamp_len} using a sliding window.')


model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True
assert model.same_length is False

skip_first = args.skip_first


###############################################################################
# Evaluation code
###############################################################################
def evaluate(eval_iter):
    global skip_first
    print(f"old behavior: {skip_first}")
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_len, total_loss = 0, 0.
    start_time = time.time()
    with torch.no_grad():
        mems = tuple()
        with tqdm(total=len(eval_iter)) as progress_bar:
            for idx, (data, target, _) in enumerate(eval_iter):
                ret = model(data, target, *mems, softmax_keep_order=True)
                bsz = target.size(1)
                loss, mems = ret[0], ret[1:]
                if skip_first:
                    # TODO add option in the model to skip computing softmax
                    # for all but the last position.
                    # shape of loss: (len,B)
                    # take only the last position, be careful with
                    # proj_adaptive_softmax which can change the indices
                    # if softmax_keep_order is False.
                    loss = loss[-1].sum(dim=-1)  # mean across batch dim
                    total_loss += loss.item()
                    total_len += bsz
                else:
                    if idx == 0:
                        total_len += loss.shape[0] * loss.shape[1]
                        loss = loss.sum()
                        total_loss += loss.item()
                    else:
                        # TODO add option in the model to skip computing
                        # softmax for all but the last position.
                        # shape of loss: (len,B)
                        # take only the last position, be careful with
                        # proj_adaptive_softmax which can change the indices
                        # if softmax_keep_order is False.
                        loss = loss[-1].sum(dim=-1)  # mean across batch dim
                        total_loss += loss.item()
                        total_len += bsz
                progress_bar.update(1)
        total_time = time.time() - start_time
    logging(f'{total_len} positions evaluated.')
    logging(f'Time : {total_time :.2f}s, '
            f'{ 1000 * total_time / (idx+1):.2f}ms/segment')
    return total_loss / total_len


# Run on test data.
if args.split == 'all':
    va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
                                  device=device,
                                  ext_len=args.ext_len, sliding_window=True)
    te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
                                  device=device,
                                  ext_len=args.ext_len, sliding_window=True)
    test_loss = evaluate(te_iter)
    valid_loss = evaluate(va_iter)
elif args.split == 'valid':
    va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len,
                                  sliding_window=True)
    valid_loss = evaluate(va_iter)
    test_loss = None
elif args.split == 'test':
    te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len,
                                  sliding_window=True)
    test_loss = evaluate(te_iter)
    valid_loss = None


def format_log(loss, split):
    if args.dataset in ['enwik8', 'text8']:
        log_str = '| {0} loss {1:5.2f} | {0} bpc {2:9.5f} '.format(
            split, loss, loss / math.log(2))
    else:
        log_str = '| {0} loss {1:5.2f} | {0} ppl {2:9.3f} '.format(
            split, loss, math.exp(loss))
    return log_str


log_str = ''
if valid_loss is not None:
    log_str += format_log(valid_loss, 'valid')
if test_loss is not None:
    log_str += format_log(test_loss, 'test')

logging('=' * 100)
logging(log_str)
logging('=' * 100)
