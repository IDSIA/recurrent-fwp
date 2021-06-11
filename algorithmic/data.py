# Dataset
import os

import numpy
import random

import torch
from torch.utils.data import Dataset


# From https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class Vocabulary(object):
    def __init__(self, vocab_dict=None, vocab_file=None,
                 include_unk=False, unk_str='<unk>',
                 include_eos=False, eos_str='<eos>',
                 no_out_str='_',
                 pad_id=None, pad_str=None):
        # If provided, contruction from dict is prioritized.
        self.str2idx = {}
        self.idx2str = []

        self.no_out_str = no_out_str

        if include_eos:
            self.add_str(eos_str)
            self.eos_str = eos_str

        if include_unk:
            self.add_str(unk_str)
            self.unk_str = unk_str

        if vocab_dict is not None:
            self.contruct_from_dict(vocab_dict)
        elif vocab_file is not None:
            self.contruct_from_file(vocab_file)

    def contruct_from_file(self, vocab_file):
        # Expect each line to contain "token_str idx", space separated.
        print(f"Creating vocab from: {vocab_file}")
        tmp_idx2str_dict = {}
        with open(vocab_file, 'r') as text:
            for line in text:
                vocab_pair = line.split()
                assert vocab_pair == 2, "Unexpected vocab format."
                token_str, token_idx = vocab_pair
        assert False, "Not implemented yet."

    def contruct_from_dict(self, vocab_dict):
        self.str2idx = vocab_dict
        vocab_size = len(vocab_dict.keys())
        assert False, "Not implemented yet."

    def get_idx(self, stg):
        return self.str2idx[stg]

    def get_str(self, idx):
        return self.idx2str(idx)

    # Increment the vocab size, give the new index to the new token.
    def add_str(self, stg):
        if stg not in self.str2idx.keys():
            self.idx2str.append(stg)
            self.str2idx[stg] = len(self.idx2str) - 1

    # Return vocab size.
    def size(self):
        return len(self.idx2str)

    def get_no_op_id(self):
        return self.str2idx[self.no_out_str]

    def get_unk_str(self):
        return self.unk_str


class LTEDataset(Dataset):

    def __init__(self, src_file, tgt_file, src_pad_idx, tgt_pad_idx,
                 src_vocab=None, tgt_vocab=None, device='cuda'):

        self.src_max_seq_length = None  # set by text_to_data
        self.tgt_max_seq_length = None

        build_src_vocab = False
        if src_vocab is None:
            build_src_vocab = True
            self.src_vocab = Vocabulary()
        else:
            self.src_vocab = src_vocab

        build_tgt_vocab = False
        if tgt_vocab is None:
            build_tgt_vocab = True
            self.tgt_vocab = Vocabulary()
        else:
            self.tgt_vocab = tgt_vocab

        self.data = self.text_to_data(
            src_file, tgt_file, src_pad_idx, tgt_pad_idx,
            build_src_vocab, build_tgt_vocab, device)

        self.data_size = len(self.data)

    def __len__(self):  # To be used by PyTorch Dataloader.
        return self.data_size

    def __getitem__(self, index):  # To be used by PyTorch Dataloader.
        return self.data[index]

    def text_to_data(self, src_file, tgt_file, src_pad_idx, tgt_pad_idx,
                     build_src_vocab=None, build_tgt_vocab=None,
                     device='cuda'):
        # Convert paired src/tgt texts into torch.tensor data.
        # All sequences are padded to the length of the longest sequence
        # of the respective file.

        assert os.path.exists(src_file)
        assert os.path.exists(tgt_file)

        data_list = []
        # Check the max length, if needed construct vocab file.
        src_max = 0
        with open(src_file, 'r') as text:
            for line in text:
                tokens = line.split()
                length = len(tokens)
                if src_max < length:
                    src_max = length
                if build_src_vocab:
                    for token in tokens:
                        self.src_vocab.add_str(token)
        self.src_max_seq_length = src_max

        tgt_max = 0
        with open(tgt_file, 'r') as text:
            for line in text:
                tokens = line.split()
                length = len(tokens)
                if tgt_max < length:
                    tgt_max = length
                if build_tgt_vocab:
                    for token in tokens:
                        self.tgt_vocab.add_str(token)
        self.tgt_max_seq_length = tgt_max

        # Construct data
        src_list = []
        print(f"Loading source file from: {src_file}")
        with open(src_file, 'r') as text:
            for line in text:
                seq = []
                tokens = line.split()
                for token in tokens:
                    seq.append(self.src_vocab.get_idx(token))
                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)
                # padding
                new_seq = var_seq.data.new(src_max).fill_(src_pad_idx)
                new_seq[:var_len] = var_seq
                src_list.append(new_seq)

        tgt_list = []
        print(f"Loading target file from: {tgt_file}")
        with open(tgt_file, 'r') as text:
            for line in text:
                seq = []
                tokens = line.split()
                for token in tokens:
                    seq.append(self.tgt_vocab.get_idx(token))

                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)
                # padding
                new_seq = var_seq.data.new(tgt_max).fill_(tgt_pad_idx)
                new_seq[:var_len] = var_seq
                tgt_list.append(new_seq)

        # src_file and tgt_file are assumed to be aligned.
        assert len(src_list) == len(tgt_list)
        for i in range(len(src_list)):
            data_list.append((src_list[i], tgt_list[i]))

        return data_list


if __name__ == '__main__':
    from datetime import datetime
    import random
    import argparse

    from torch.utils.data import DataLoader

    torch.manual_seed(123)
    random.seed(123)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    parser = argparse.ArgumentParser(description='Learning to execute')
    parser.add_argument(
        '--data_dir', type=str,
        default='./data/',
        help='location of the data corpus')

    args = parser.parse_args()
    data_path = args.data_dir

    file_src = f"{data_path}/valid_3.src"
    file_tgt = f"{data_path}/valid_3.tgt"

    bsz = 3

    dummy_data = LTEDataset(src_file=file_src, tgt_file=file_tgt,
                            src_pad_idx=0, tgt_pad_idx=-1,
                            src_vocab=None, tgt_vocab=None)

    data_loader = DataLoader(dataset=dummy_data, batch_size=bsz, shuffle=True)

    stop_ = 2

    for idx, batch in enumerate(data_loader):
        src, tgt = batch
        if idx < stop_:
            print(src[:, 0:20])
