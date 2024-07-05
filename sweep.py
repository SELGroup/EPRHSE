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
    # args.beta_pool=wandb.config.beta_pool
    # args.pre_lr=wandb.config.pre_lr
    # args.lr=wandb.config.lr
    args.regs= f'[{wandb.config.regs1}, 1e-7]'
    args.hgcn_mix = f'[{wandb.config.hgcn_mix1}, {wandb.config.hgcn_mix2}]'
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

# sweep_config={
#     "method":"random",
#     "name":"xmrec_cn",
#     "metric":{"goal":"maximize","name":"score"},
#     "parameters":{
#         "beta_pool":{"max":0.95,"min":0.01},
#         "pre_lr":{"values":[0.001,0.005,0.01,0.05,0.1,0.5]},
#         "lr":{"values":[0.001,0.005,0.01,0.05,0.1,0.5]},
#         "regs1":{"values":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]},
#         "hgcn_mix1":{"values":[1,2,5,8,10]},
#         "hgcn_mix2":{"values":[1,1e-1,1e-2,1e-3]},
#         }
#     }

# sweep_id=wandb.sweep(sweep=sweep_config,project="UPRTH")
# wandb.agent(sweep_id, function=main, count=100)