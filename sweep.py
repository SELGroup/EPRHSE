import wandb
from main import train
import torch
import random
import numpy as np
import dgl
from utility.parser import *

def main():
    wandb.init(project="UPRTH")
    args = parse_args()
    args.regs= f'[{wandb.config.regs1}, {wandb.config.regs2}]'
    args.hgcn_mix = f'[{wandb.config.hgcn_mix1}, {wandb.config.hgcn_mix2}]'
    args.beta_pool=args.beta_pool*1.0/100
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True #使得网络相同输入下每次运行的输出固定
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    dgl.seed(args.random_seed)

    rec1,rec2,ndcg1,ndcg2=train(args)
    wandb.log({"rec1":rec1,
               "rec2":rec2,
               "ndcg1":ndcg1,
               "ndcg2":ndcg2,
               "score":rec1+rec2+ndcg1+ndcg2})
    
    wandb.finish()

if __name__ == '__main__':
    main()