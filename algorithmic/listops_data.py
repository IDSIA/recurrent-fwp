# ListOps Dataset
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


class ListOpsVocabulary(object):
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

    def print_map(self):
        for i in range(len(self.idx2str)):
            print(f"{i} : {self.idx2str[i]}")

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


class ListOpsDataset(Dataset):

    def __init__(self, data_file, pad_idx=0, max_out_val=10, vocab=None,
                 device='cuda'):

        self.max_seq_length = None  # set by text_to_data

        build_vocab = False
        if vocab is None:
            build_vocab = True
            self.vocab = ListOpsVocabulary()
        else:
            self.vocab = vocab

        self.data = self.text_to_data(data_file, pad_idx, build_vocab,
                                      max_out_val, device)

        self.data_size = len(self.data)

    def __len__(self):  # To be used by PyTorch Dataloader.
        return self.data_size

    def __getitem__(self, index):  # To be used by PyTorch Dataloader.
        return self.data[index]

    def text_to_data(self, data_txt, pad_idx, build_vocab=False,
                     max_out_val=10, device='cuda'):
        # The data file is expected to be formatted as:
        # SEQ_LEN TARGET INPUT:
        # Convert it into torch.tensor data.
        # All sequences are padded to the length of the longest sequence
        # of the respective file (lazy padding).

        assert os.path.exists(data_txt)

        data_list = []
        # Check the max length, if needed construct vocab file.
        max_len = 0
        with open(data_txt, 'r') as text:
            for line in text:
                # The data file is expected to be formatted as:
                # SEQ_LEN TARGET INPUT...
                tokens = line.split()
                length = int(tokens[0])

                if max_len < length:
                    max_len = length

                assert -1 < int(tokens[1]) < max_out_val

                if build_vocab:
                    for token in tokens[2:]:
                        self.vocab.add_str(token)

        self.max_seq_length = max_len

        # Construct data
        input_data_list = []
        target_data_list = []
        length_data_list = []

        print(f"Loading data file from: {data_txt}")
        with open(data_txt, 'r') as text:
            for line in text:
                seq = []
                tokens = line.split()

                # length
                # minus one to directly store index
                length_val = torch.tensor(
                    int(tokens[0]) - 1, device=device, dtype=torch.int64)
                length_data_list.append(length_val)

                # target
                tgt_val = torch.tensor(
                    int(tokens[1]), device=device, dtype=torch.int64)
                target_data_list.append(tgt_val)

                # input seq
                for token in tokens[2:]:
                    seq.append(self.vocab.get_idx(token))
                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)
                # padding
                new_seq = var_seq.data.new(max_len).fill_(pad_idx)
                new_seq[:var_len] = var_seq
                input_data_list.append(new_seq)

        # src_file and tgt_file are assumed to be aligned.
        assert len(input_data_list) == len(target_data_list)
        assert len(input_data_list) == len(length_data_list)

        for i in range(len(input_data_list)):
            data_list.append(
                (length_data_list[i], target_data_list[i], input_data_list[i]))

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
    # parser = argparse.ArgumentParser(description='Learning to execute')
    # parser.add_argument(
    #     '--data_dir', type=str,
    #     default='./data/',
    #     help='location of the data corpus')

    # args = parser.parse_args()
    # data_path = args.data_dir
    data_file = "utils/listops_data/valid_a10d10max201min19.txt"

    bsz = 3

    dummy_data = ListOpsDataset(data_file, pad_idx=0)
    dummy_data.vocab.print_map()

    data_loader = DataLoader(dataset=dummy_data, batch_size=bsz, shuffle=True)

    stop_ = 2

    for idx, batch in enumerate(data_loader):
        seq_len, labels, inputs = batch
        if idx < stop_:
            print(seq_len)
            print(labels)
            print(inputs)
