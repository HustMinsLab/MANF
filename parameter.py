# -*- coding: utf-8 -*-

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='IDRR')
    
    # for word2vec
    parser.add_argument('--WORD2VEC_DIR', default='word2vec_0.05.pickle')
    
    # dataset
    parser.add_argument('--train_size', default=17945,  type=int,   help='Train set size')
    parser.add_argument('--dev_size',   default=1653,   type=int,   help='Dev set size')
    parser.add_argument('--test_size',  default=1474,   type=int,   help='Test set size')
    parser.add_argument('--num_class',  default=4,      type=int,   help='Number of class')

    # # model arguments
    parser.add_argument('--in_dim',     default=300,   type=int,   help='Size of input word vector')
    parser.add_argument('--h_dim',      default=300,   type=int,   help='Size of hidden unit')
    parser.add_argument('--len_arg',    default=50,    type=int,   help='Argument length')

    # # training arguments
    parser.add_argument('--seed',       default=209,    type=int,   help='seed for reproducibility')    
    parser.add_argument('--batch_size', default=32,     type=int,   help='batchsize for optimizer updates')
    parser.add_argument('--wd',         default=1e-5,   type=float, help='weight decay')
    parser.add_argument('--momentum',   default=0.9,    type=float)
    parser.add_argument('--num_epoch',  default=15,     type=int,   help='number of total epochs to run') 
    parser.add_argument('--lr',         default=5e-4,  type=float,  help='initial learning rate')
    parser.add_argument('--stepLR',     default=1,      type=int,   help='1 for open, 0 for closed')
    parser.add_argument('--milestones', default=4,      type=int,   help='step that lr decays')
    parser.add_argument('--gamma',      default=0.2,    type=float, help='rate of lr decays')
        
    args = parser.parse_args()
    return args
