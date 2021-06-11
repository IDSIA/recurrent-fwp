# Code to generate the data set for ListOps task.
#
# Fork from the original ListOps paper: https://github.com/nyu-mll/spinn/blob/master/python/spinn/data/listops/make_data.py
# See LICENSE file for the license.
#
# Main modifications:
# - no extra opening parentheses.
# - no depth singleton sequence
# - minimum and maximum sequence length options (like LRA version)
# see also https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/data/listops.py

import random
import numpy as np

MIN = "[MIN"
MAX = "[MAX"
MED = "[MED"  # int med
FIRST = "[FIRST"
LAST = "[LAST"
SUM_MOD = "[SM"  # sum modulo 10
END = "]"

OPERATORS = [MIN, MAX, FIRST]  # , FIRST, LAST]
VALUES = range(10)

VALUE_P = 0.25
MAX_ARGS = 5  # LRA 10, NYU: 5
MAX_DEPTH = 20  # LRA 10, NYU: 20

MAX_LENGTH = 401  # counted in terms of characters.
# Basically no min, as there is at least one op (singleton excluded)
# MIN_LENGTH = 19
MIN_LENGTH = 299


def generate_tree(depth):
    if depth < MAX_DEPTH:
        r = random.random()
    else:  # if max depth is reached, just output a value. No more ops.
        r = 1

    # change from the original code: avoid singleton list.
    if r > VALUE_P and depth > 1:  # with VALUE_P (here 25%) chance, just output a value.
        value = random.choice(VALUES)
        return value, 1, depth
    else:
        length = 2  # op and closing bracket
        num_values = random.randint(2, MAX_ARGS)  # draw number of args
        values = []
        deepest = 1
        for _ in range(num_values):
            sub_tree, arg_len, sub_depth = generate_tree(depth + 1)
            values.append(sub_tree)  # each args can also be ops.
            length += arg_len
            if sub_depth > deepest:
                deepest = sub_depth

        op = random.choice(OPERATORS)  # draw op
        t = (op, values[0])
        for value in values[1:]:
            t = (t, value)
        t = (t, END)
    return t, length, deepest


def to_string(t, parens=True):
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        if parens:
            return '( ' + to_string(t[0]) + ' ' + to_string(t[1]) + ' )'
        else:  # missing in the original code
            return to_string(t[0], parens) + ' ' + to_string(t[1], parens)


def to_value(t):
    if not isinstance(t, tuple):
        return t
    l = to_value(t[0])
    r = to_value(t[1])
    if l in OPERATORS:  # Create an unsaturated function.
        return (l, [r])
    elif r == END:  # l must be an unsaturated function.
        if l[0] == MIN:
            return min(l[1])
        elif l[0] == MAX:
            return max(l[1])
        elif l[0] == FIRST:
            return l[1][0]
        elif l[0] == LAST:
            return l[1][-1]
        elif l[0] == MED:
            return int(np.median(l[1]))
        elif l[0] == SUM_MOD:
            return (np.sum(l[1]) % 10)
    elif isinstance(l, tuple):  # We've hit an unsaturated function and an argument.
        return (l[0], l[1] + [r])


if __name__ == '__main__':

    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Generate data.')
    parser.add_argument('--dump_dir',
        required=True, help='where to store the data')
    parser.add_argument('--train_size', required=False, default=10000,
        type=int, help='Number of examples in the train set.')
    parser.add_argument('--valid_size', required=False, default=1000,
        type=int, help='Number of examples in the valid set.')
    parser.add_argument('--test_size', required=False, default=1000,
        type=int, help='Number of examples in the test set.')
    parser.add_argument('--only_depth', required=False, default=0,
        type=int, help='If positive, only select sequence with this depth.')

    args = parser.parse_args()

    rnd_seed = 42
    random.seed(rnd_seed)

    tr_file = f"train_a{MAX_ARGS}d{MAX_DEPTH}max{MAX_LENGTH}min{MIN_LENGTH}dep{args.only_depth}"
    valid_file = f"valid_a{MAX_ARGS}d{MAX_DEPTH}max{MAX_LENGTH}min{MIN_LENGTH}dep{args.only_depth}"
    test_file = f"test_a{MAX_ARGS}d{MAX_DEPTH}max{MAX_LENGTH}min{MIN_LENGTH}dep{args.only_depth}"

    train_txt = f"{args.dump_dir}/{tr_file}.txt"
    valid_txt = f"{args.dump_dir}/{valid_file}.txt"
    test_txt = f"{args.dump_dir}/{test_file}.txt"

    train_size = args.train_size
    valid_size = args.valid_size
    test_size = args.test_size

    # train set
    prog_bar = tqdm(total=train_size)
    seq_count = 0
    len_stat = []
    with open(train_txt, 'a') as txt_in:
        while seq_count < train_size:
            tree, length, depth = generate_tree(1)
            if args.only_depth > 0:
                if depth == args.only_depth:
                    len_stat.append(length)
                    data_str = to_string(tree, parens=False)
                    data_tgt = to_value(tree)
                    seq_str = f"{length} {data_tgt} {data_str}"
                    txt_in.write(seq_str + '\n')
                    prog_bar.update(1)
                    seq_count += 1

#            # To filter based on min/max length, use:
#            if length < MAX_LENGTH and length > MIN_LENGTH:
#                len_stat.append(length)
#                data_str = to_string(tree, parens=False)
#                data_tgt = to_value(tree)
#                seq_str = f"{length} {data_tgt} {data_str}"
#                txt_in.write(seq_str + '\n')
#                prog_bar.update(1)
#                seq_count += 1

    print(f"train mean length: {np.mean(len_stat)}, "
          f"median: {np.median(len_stat)}, "
          f"max: {np.max(len_stat)}, min: {np.min(len_stat)}")

    # valid set
    prog_bar = tqdm(total=valid_size)
    seq_count = 0
    len_stat = []
    with open(valid_txt, 'a') as txt_in:
        while seq_count < valid_size:
            tree, length, depth = generate_tree(1)
            if args.only_depth > 0:
                if depth == args.only_depth:
                    len_stat.append(length)
                    data_str = to_string(tree, parens=False)
                    data_tgt = to_value(tree)
                    seq_str = f"{length} {data_tgt} {data_str}"
                    txt_in.write(seq_str + '\n')
                    prog_bar.update(1)
                    seq_count += 1

#            if length < MAX_LENGTH and length > MIN_LENGTH:
#                len_stat.append(length)
#                data_str = to_string(tree, parens=False)
#                data_tgt = to_value(tree)
#                seq_str = f"{length} {data_tgt} {data_str}"
#                txt_in.write(seq_str + '\n')
#                prog_bar.update(1)
#                seq_count += 1

    print(f"valid mean length: {np.mean(len_stat)}, "
          f"median: {np.median(len_stat)}, "
          f"max: {np.max(len_stat)}, min: {np.min(len_stat)}")

    # test set
    prog_bar = tqdm(total=test_size)
    seq_count = 0
    len_stat = []
    with open(test_txt, 'a') as txt_in:
        while seq_count < test_size:
            tree, length, depth = generate_tree(1)
            if args.only_depth > 0:
                if depth == args.only_depth:
                    len_stat.append(length)
                    data_str = to_string(tree, parens=False)
                    data_tgt = to_value(tree)
                    seq_str = f"{length} {data_tgt} {data_str}"
                    txt_in.write(seq_str + '\n')
                    prog_bar.update(1)
                    seq_count += 1

#            if length < MAX_LENGTH and length > MIN_LENGTH:
#                len_stat.append(length)
#                data_str = to_string(tree, parens=False)
#                data_tgt = to_value(tree)
#                seq_str = f"{length} {data_tgt} {data_str}"
#                txt_in.write(seq_str + '\n')
#                prog_bar.update(1)
#                seq_count += 1

    print(f"test mean length: {np.mean(len_stat)}, "
          f"median: {np.median(len_stat)}, "
          f"max: {np.max(len_stat)}, min: {np.min(len_stat)}")
