from ast import Lambda
import random
import numpy as np
import torch
from args import get_args

import utils.config_util as cu
from utils.logging import get_logger
from tasks.run import standard_run
from itertools import product

if __name__ == '__main__':
    logger = get_logger()
    args = get_args()

    algorithm = args.algorithm
    masking_ratio = float(args.masking_ratio)
    anneal_time = args.anneal_time
    t_max = args.t_max
    minigame = args.minimap
    seed = args.seed
    momentum = float(args.momentum)

    config = cu.config_copy(cu.get_config(algorithm, minigame, masking_ratio, anneal_time, t_max, seed,momentum))
    print(f'{minigame}_{seed}_{algorithm} start ')
    np.random.seed(seed)
    torch.manual_seed(seed)
    config['env_args']['seed'] = seed    
    standard_run(config, logger, minigame)

