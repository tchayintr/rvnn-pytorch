# Vanilla Recursive Neural Network Implementation using PyTorch

Original Paper: 
 - https://www.aclweb.org/anthology/D13-1170/
 - https://www.aclweb.org/anthology/D11-1014/

Reference: 
 - https://github.com/aykutfirat/pyTorchTree

### Requirements

 - nltk
 - pytorch
 - progressbar

### How to run (cmd.sh)

`$ python src/main.py --train_data data/sstb/train.txt --valid_data data/sstb/valid.txt --test_data data/sstb/test.txt --vocab_indices models/train.vocab --cuda --epoch 10 --embed_dim 100 --num_class 5 --max_norm 5 --norm_type 2 --learning_rate 0.1 --sgd_momentum_ratio 0.9`

#### Arguments description

`$ python src/main.py -h`
