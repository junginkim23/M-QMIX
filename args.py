import os
import argparse
import torch

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--minimap', default='27m_vs_30m', type=str)
    parser.add_argument('--masking-ratio', default=0.2, help='0.2')
    parser.add_argument('--algorithm', default='RNN_AGENT/qmix_ssl_beta', type=str)
    parser.add_argument('--t_max', default=2500000, type=int)
    parser.add_argument('--anneal_time', default=50000, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--momentum', default=0.999, type=float)

    args = parser.parse_args()

    return args
