# Recurrent Fast Weight Programmers

This is the official repository containing the code we used to produce the experimental results reported in the paper:

[Going Beyond Linear Transformers with Recurrent Fast Weight Programmers (NeurIPS 2021)](https://arxiv.org/abs/2106.06295)

## Contents

* `algorithmic` directory for code execution and ListOps
* `language_modeling` directory for language modeling
* `reinforcement_learning` directory for RL

Separate license files can be found under each directory.

## General instructions
Please refer to the readme file in each directory for further instructions.

In all tasks, our custom CUDA kernels will be automatically compiled.
To avoid recompiling the code multiple times, we recommend to specify the path to a directory to store the compiled code via:
```
export TORCH_EXTENSIONS_DIR="/home/me/torch_extensions/lm"
```
Such a line is already included in the example scripts we provide. Please change the path to a safe directory of your choice.

Important: separate paths should be used for different tasks (i.e. here, one for language modeling, one for code execution, one for ListOps, and one for RL).

## BibTex
```
@inproceedings{irie2021going,
      title={Going Beyond Linear Transformers with Recurrent Fast Weight Programmers}, 
      author={Kazuki Irie and Imanol Schlag and R\'obert Csord\'as and J\"urgen Schmidhuber},
      booktitle={Proc. Advances in Neural Information Processing Systems (NeurIPS)},
      address={Virtual only},
      year={2021}
}
```

## Links
* This is a follow up on our previous work: [Linear Transformers are Secretly Fast Weight Programmers (ICML 2021)](https://arxiv.org/abs/2102.11174)
* [Jürgen Schmidhuber's AI blog post on Fast Weight Programmers (March 26, 2021)](https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html).

