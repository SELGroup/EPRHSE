import sys

import torch
import torch.nn as nn
import numpy as np
import sklearn
import torch.nn.functional as F
import dgl
import numpy as np
from utility.LightGCNLayer import LightGCNLayer
from utility.AttLayer import MultiHeadGATLayer
import dgl.function as fn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from typing import Callable
from torch_geometric.typing import Adj, Size
from torch import Tensor
from torch_sparse import SparseTensor, matmul

#import torchmetrics

'''TODO: construct prostive and negative item-bT_idx/cpr_idx bipartite graph for bpr loss'''


def construct_user_item_bigraph(graph):
    return graph.node_type_subgraph(['user', 'item'])  #创建user-item子图


def construct_item_related_bigraph(graph, node_type='bT_idx'):
    return graph.node_type_subgraph([node_type, 'item'])


def construct_user_related_bigraph(graph, node_type='age'):
    return graph.node_type_subgraph([node_type, 'user'])


def construct_negative_item_graph(graph, k, device, node_type='bT_idx'):
    edge_dict = {'bT_idx': ['bi', 'ib'], 'cpr_idx': ['pi', 'ip']}
    user_item_src, user_item_dst = graph.edges(etype=edge_dict[node_type][0])
    neg_src = user_item_src.repeat_interleave(k)
    n_neg_src = len(user_item_src)
    neg_dst = torch.randint(0, graph.num_nodes(ntype='item'), (n_neg_src * k,)).to(device)
    data_dict = {
        (node_type, edge_dict[node_type][0], 'item'): (neg_src, neg_dst),
        ('item', edge_dict[node_type][1], node_type): (neg_dst, neg_src)
    }
    num_dict = {
        node_type: graph.num_nodes(ntype=node_type), 'item': graph.num_nodes(ntype='item'),
    }
    return dgl.heterograph(data_dict, num_nodes_dict=num_dict)


def construct_negative_user_graph(graph, k, device, node_type='age'):
    edge_dict = {'age': ['au', 'ua'], 'job': ['ju', 'uj']}
    user_item_src, user_item_dst = graph.edges(etype=edge_dict[node_type][0])
    neg_src = user_item_src.repeat_interleave(k)
    n_neg_src = len(user_item_src)
    neg_dst = torch.randint(0, graph.num_nodes(ntype='user'), (n_neg_src * k,)).to(device)
    data_dict = {
        (node_type, edge_dict[node_type][0], 'user'): (neg_src, neg_dst),
        ('user', edge_dict[node_type][1], node_type): (neg_dst, neg_src)
    }
    num_dict = {
        node_type: graph.num_nodes(ntype=node_type), 'user': graph.num_nodes(ntype='user'),
    }
    return dgl.heterograph(data_dict, num_nodes_dict=num_dict)


def arr_remove_idx(A, del_idx):  # return a 1-D array
    # print(A)
    # print(del_idx)
    mask = np.ones(A.shape, dtype=bool)
    mask[range(A.shape[0]), del_idx] = False
    output = A[mask]
    return output


def construct_negative_item_graph_c2ep(graph, n_type, device, node_type='cate'):  # n_cate = 12
    edge_dict = {'cate': ['ci', 'ic'], 'rate': ['ri', 'ir']}
    range_list = {'cate': np.arange(0, n_type, 1), 'rate': np.arange(0, n_type, 1)}  #[0,1,2,...,n_type-1]
    user_item_src, user_item_dst = graph.edges(etype=edge_dict[node_type][1])
    # print(user_item_src.type())
    # print(user_item_dst[:40])
    neg_src = user_item_src.repeat_interleave(range_list[node_type].shape[0] - 1)
    neg_dst = np.array([range_list[node_type], ] * len(user_item_src))  #len(user_item_src) * n_type 大小的二维数组，所有src都与每个dst相连
    neg_dst = torch.from_numpy(arr_remove_idx(neg_dst, np.array(user_item_dst.cpu()))).to(device)  #移除原始图中存在的dst，并转为tensor
    data_dict = {
        (node_type, edge_dict[node_type][0], 'item'): (neg_dst, neg_src),
        ('item', edge_dict[node_type][1], node_type): (neg_src, neg_dst)
    }
    num_dict = {
        node_type: graph.num_nodes(ntype=node_type), 'item': graph.num_nodes(ntype='item'),
    }
    return dgl.heterograph(data_dict, num_nodes_dict=num_dict)


def construct_negative_user_graph_c2ep(graph, n_type, device, node_type='age'):  # n_cate = 12
    edge_dict = {'age': ['au', 'ua'], 'job': ['ju', 'uj']}
    range_list = {'age': np.arange(0, n_type, 1), 'job': np.arange(0, n_type, 1)}
    user_item_src, user_item_dst = graph.edges(etype=edge_dict[node_type][1])
    # print(user_item_src.type())
    # print(user_item_dst[:40])
    neg_src = user_item_src.repeat_interleave(range_list[node_type].shape[0] - 1)
    neg_dst = np.array([range_list[node_type], ] * len(user_item_src))
    neg_dst = torch.from_numpy(arr_remove_idx(neg_dst, np.array(user_item_dst.cpu()))).to(device)
    data_dict = {
        (node_type, edge_dict[node_type][0], 'user'): (neg_dst, neg_src),
        ('user', edge_dict[node_type][1], node_type): (neg_src, neg_dst)
    }
    num_dict = {
        node_type: graph.num_nodes(ntype=node_type), 'user': graph.num_nodes(ntype='user'),
    }
    return dgl.heterograph(data_dict, num_nodes_dict=num_dict)


def construct_negative_graph(graph, k, device):
    user_item_src, user_item_dst = graph.edges(etype='ui')
    neg_src = user_item_src.repeat_interleave(k)  #将原始元素重复k次，二部图的噪声边是原始k倍
    n_neg_src = len(user_item_src)
    neg_dst = torch.randint(0, graph.num_nodes(ntype='item'), (n_neg_src * k,)).to(device)  #随机噪声
    data_dict = {
        ('user', 'ui', 'item'): (neg_src, neg_dst),
        ('item', 'iu', 'user'): (neg_dst, neg_src),
        # ('neg_user', 'ui', 'side'): (user_side_src, user_side_dst),
        # ('side', 'iu', 'neg_user'): (user_side_dst, user_side_src)
    }
    num_dict = {
        'user': graph.num_nodes(ntype='user'), 'item': graph.num_nodes(ntype='item'),
        # 'side': graph.num_nodes(ntype='side')
    }
    return dgl.heterograph(data_dict, num_nodes_dict=num_dict)


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x): #用两端节点向量内积表示边的score
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']

    def alignment_forward(self, edge_subgraph, x):  #用两端节点向量差表示边的score（特征向量形式），对齐得分
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_sub_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']


class HGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)
        self.fc_2 = nn.LazyLinear(64)
        self.d_sqrt = 8
        self.edgedrop = dgl.transforms.DropEdge(0.2) #随机删除20%的边（减少过拟合风险）或模拟数据缺失
        self.leakyrelu = nn.LeakyReLU(0.1) #激活函数
        # self.dropout = nn.Dropout(p=0.01)

    def forward(self, graph, h, etype_forward, etype_back, norm_2=-1, alpha=0, pretrained_feature=None, detach=False,
                att=0):
        with graph.local_scope():
            src, _, dst = etype_forward
            feat_src = h[src]
            feat_dst = h[dst + '_edge']
            aggregate_fn = fn.copy_u('h', 'm')#从源节点复制特征'h'到消息'm'中
            aggregate_fn_back = fn.copy_u('h_b', 'm_b')
            # aggregate_fn = fn.copy_src('h', 'm')
            # aggregate_fn_back = fn.copy_src('h_b', 'm_b')

            graph.nodes[src].data['h'] = feat_src
            # graph.nodes[src].data['h'] = self.dropout(feat_src)
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'), etype=etype_forward)  #消息聚合函数，消息求和，消息传递的边类型

            rst = graph.nodes[dst].data['h'] #聚合后目标节点的消息/特征
            in_degrees = graph.in_degrees(etype=etype_forward).float().clamp(min=1) #etype_forward边类型下节点入度
            norm_dst = torch.pow(in_degrees, -1)
            shp_dst = norm_dst.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm_dst, shp_dst)
            if alpha != 0:
                if pretrained_feature != None:
                    if detach == True:
                        pretrained_feature = pretrained_feature.detach().requires_grad_(False)  #分离张量，梯度设置不可导
                    if att != 0:
                        if att == 1:
                            '''attention calculation by DGL'''
                            graph.nodes[dst].data['q'] = torch.unsqueeze(rst, dim=1)  #原先的n*d变成了n*1*d，每一行变成了一个单独的矩阵
                            graph.nodes[dst].data['k'] = torch.transpose(pretrained_feature, 1, 2)  #第一维和第二维转置n*t*d->n*d*t
                            graph.apply_nodes(lambda nodes: {'a': torch.matmul(
                                F.softmax(torch.matmul(nodes.data['q'], nodes.data['k']) / self.d_sqrt, dim=2),  #预训练feature的权重  n*1*t
                                pretrained_feature).squeeze()}, ntype=dst)  #(n*1*t)*(n*t*d)->(n*1*d)   对目标节点的操作
                            # print(graph.nodes[dst].data['a'] .shape)
                            pretrained_feature = graph.nodes[dst].data['a']
                        elif att == 2:
                            pretrained_feature = self.fc_2(pretrained_feature)
                    rst = rst + alpha * torch.tanh(pretrained_feature)  #目标节点聚合特征+加权的预训练目标节点特征
                    # rst = alpha * torch.tanh(pretrained_feature)
                    # rst = rst + torch.softmax(pretrained_feature,dim=1)
                    # rst = rst + self.leakyrelu(pretrained_feature)
                else:
                    rst = rst + alpha * feat_dst  #没有预训练的feature则加自身

            rst = rst * norm #消息聚合后，目标节点特征归一化
            graph.nodes[dst].data['h_b'] = rst
            graph.update_all(aggregate_fn_back, fn.sum(msg='m_b', out='h_b'), etype=etype_back)  #从目标节点反向聚合消息到源节点
            bsrc = graph.nodes[src].data['h_b']

            in_degrees_b = graph.in_degrees(etype=etype_back).float().clamp(min=1)  #etype_back类型边下节点的入度, clamp设置最小为1
            norm_src = torch.pow(in_degrees_b, norm_2)  #(n,)
            shp_src = norm_src.shape + (1,) * (feat_src.dim() - 1)  #(n,) + (1,) = (n,1)
            norm_src = torch.reshape(norm_src, shp_src)  #(n,1)
            bsrc = bsrc * norm_src  #源节点特征归一化  #(n,d) 与 (n,1)-广播对应元素相乘
            #这一步实质将聚合的取平均norm_2=-1

            return bsrc, rst

class SEPooling(MessagePassing):
    def __init__(self, nn: Callable=None, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, edge_index: Adj, size: Size = None) -> Tensor:
        out = self.propagate(edge_index, x=x, size=size)
        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

class SEP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim=dim
        self.nn = nn.Sequential(
            #nn.Linear(dim, dim),
            nn.ReLU(),
            #nn.BatchNorm1d(dim),
        )

    
    def forward(self, graph, h, etype_forward, norm_2=-1):
        with graph.local_scope():
            src, _, dst = etype_forward  #节点类型
            feat_src = h
            feat_dst = torch.empty((graph.nodes(dst).shape[0], self.dim), requires_grad=False)
            #消息函数
            aggregate_fn = fn.copy_u('h', 'm')
            graph.nodes[src].data['h'] = feat_src
            #聚合函数
            reduce_fn = fn.sum(msg='m', out='h')
            #更新函数
            graph.update_all(aggregate_fn,reduce_fn,etype=etype_forward)

            rst = graph.nodes[dst].data['h']
            #归一化
            in_degrees = graph.in_degrees(etype=etype_forward).float().clamp(min=1) #etype_forward边类型下节点入度
            norm_dst = torch.pow(in_degrees, norm_2)
            shp_dst = norm_dst.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm_dst, shp_dst)
            rst = rst * norm

            return self.nn(rst)  #是否需要激活函数？


    
class SEP_U(torch.nn.Module):
    def __init__(self, embed_size, beta, device, num_blocks=2):
        super(SEP_U, self).__init__()
        self.dim = embed_size  
        self.num_blocks=num_blocks
        self.HGCN = HGCNLayer()
        self.HGCNLayers = self.get_HGCNs()  #一个还是多个HGCNLayer？
        self.sep = SEP(self.dim)
        self.sepools = self.get_sepool()  #一个还是多个SEP？
        self.device=device
        self.fc = nn.Linear(2*embed_size,embed_size)
        self.beta = beta

    def get_HGCNs(self):
        HGCNs = nn.ModuleList()  #管理模型层的容器
        for _ in range(self.num_blocks*2 + 1):
            hgcn = HGCNLayer()
            HGCNs.append(hgcn)
        return HGCNs
    
    def get_sepool(self):
        pools = nn.ModuleList()  #管理模型层的容器
        for _ in range(self.num_blocks*2):
            '''
            pool = SEPooling(
                nn.Sequential(
                    #nn.Linear(self.dim, self.dim),
                    nn.ReLU(),
                    #nn.BatchNorm1d(self.dim),
                ))
            '''
            pool = SEP(self.dim)
            pools.append(pool)
        return pools

    def forward(self, partition, graph, h0, etype_forward, etype_back, norm_2):
        h_save=[]
        src_type, _, dst_type = etype_forward
        h=h0
        #创建节点与超边的二部图
        g0=graph.node_type_subgraph([src_type, dst_type]) #创建子图
        data_dict = {
            etype_forward: (partition[0][1].cpu(), partition[0][2].cpu()), #partition[][1] partition[][2]是新的二部图的src dst对
            etype_back: (partition[0][2].cpu(), partition[0][1].cpu()),
        }
        num_dict = {
            src_type: max(partition[0][0].cpu())+1, dst_type: max(partition[0][2].cpu())+1,
        }
        g1=dgl.heterograph(data_dict, num_nodes_dict=num_dict).to(self.device)
        data_dict = {
            etype_forward: (partition[1][1].cpu(), partition[1][2].cpu()), #partition[][1] partition[][2]是新的二部图的src dst对
            etype_back: (partition[1][2].cpu(), partition[1][1].cpu()),
        }
        num_dict = {
            src_type: max(partition[1][0].cpu())+1, dst_type: max(partition[1][2].cpu())+1,
        }
        g2=dgl.heterograph(data_dict, num_nodes_dict=num_dict).to(self.device)
        g_list=[g0,g1,g2]  
        #创建节点和社区的二部图
        data_dict = {
            ('node', 'nc', 'comm'): (list(range(len(partition[0][0]))), partition[0][0].cpu()),
            ('comm', 'cn', 'node'): (partition[0][0].cpu(), list(range(len(partition[0][0])))),  #partition[][0]为社区分配list
        }
        num_dict = {
            'node': len(partition[0][0]), 'comm': max(partition[0][0].cpu())+1,
        }
        g1_comm=dgl.heterograph(data_dict, num_nodes_dict=num_dict).to(self.device)
        data_dict = {
            ('node', 'nc', 'comm'): (list(range(len(partition[1][0]))), partition[1][0].cpu()),
            ('comm', 'cn', 'node'): (partition[1][0].cpu(), list(range(len(partition[1][0])))),
        }
        num_dict = {
            'node': len(partition[1][0]), 'comm': max(partition[1][0].cpu())+1,
        }
        g2_comm=dgl.heterograph(data_dict, num_nodes_dict=num_dict).to(self.device)
        g_comm_list=[g1_comm, g2_comm]
        #edge_index0=torch.tensor([list(range(len(partition[0][0]))), partition[0][0]]).to(self.device) #第一行节点，第二行社区。partition[_][0]是list S,S[i]为用户i分到的社区
        #edge_index1=torch.tensor([list(range(len(partition[1][0]))), partition[1][0]]).to(self.device)
        #edge_index_list=[edge_index0,edge_index1]

        #down sampling
        for _ in range(self.num_blocks): #_取0,1
            #HGCN
            #h_src, h_dst = self.HGCNLayers[_](g_list[_], h, etype_forward, etype_back, norm_2)
            h_src, h_dst = self.HGCN(g_list[_], h, etype_forward, etype_back, norm_2)
            h_save.append(h_src)
            #SEP
            #h_src = self.sepools[_](g_comm_list[_], h_src, ('node', 'nc', 'comm'))
            h_src = self.sep(g_comm_list[_], h_src, ('node', 'nc', 'comm'))
            h={src_type: h_src, dst_type + '_edge': h_dst}
        #up sampling
        for _ in range(self.num_blocks,0,-1): #_取2，1
            #HGCN
            #h_src, h_dst=self.HGCNLayers[self.num_blocks*2-_](g_list[_], h, etype_forward, etype_back, norm_2)
            h_src, h_dst=self.HGCN(g_list[_], h, etype_forward, etype_back, norm_2)
            #SEP-U
            #h_src = self.sepools[self.num_blocks*2-_](g_comm_list[_-1], h_src, ('comm', 'cn', 'node'))
            h_src = self.sep(g_comm_list[_-1], h_src, ('comm', 'cn', 'node'))
            #h_src=self.fc(torch.cat([h_src,h_save[_-1]], dim=1))
            h_src=self.beta*h_src+h_save[_-1]
            h={src_type: h_src, dst_type + '_edge': h_dst}
        #last HGCN
        #h_src, h_dst=self.HGCNLayers[-1](g_list[0], h, etype_forward, etype_back, norm_2)
        h_src, h_dst=self.HGCN(g_list[0], h, etype_forward, etype_back, norm_2)

        return h_src, h_dst


class HGCNLayer_general(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, h, etype_list, norm_2=-1):
        with graph.local_scope():
            aggregate_fn = fn.copy_u('h', 'm')
            aggregate_fn_back = fn.copy_u('h_b', 'm_b')
            # aggregate_fn = fn.copy_src('h', 'm')
            # aggregate_fn_back = fn.copy_src('h_b', 'm_b')
            for etype in etype_list:
                etype_forward, _ = etype
                src, _, dst = etype_forward
                feat_src = h[src]
                feat_dst = h[dst]

                graph.nodes[src].data['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'), etype=etype_forward)

                rst = graph.nodes[dst].data['h']
                in_degrees = graph.in_degrees(etype=etype_forward).float().clamp(min=1)
                norm_dst = torch.pow(in_degrees, -1)
                shp_dst = norm_dst.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm_dst, shp_dst)
                rst = rst * norm
                graph.nodes[dst].data['h_b'] = rst

            update_dict = {}
            in_degrees_b = None
            for etype in etype_list:
                _, etype_back = etype
                update_dict[etype_back] = (aggregate_fn_back, fn.sum(msg='m_b', out='h_b'))
                if in_degrees_b == None:
                    in_degrees_b = graph.in_degrees(etype=etype_back).float().clamp(min=1)
                else:
                    in_degrees_b += graph.in_degrees(etype=etype_back).float().clamp(min=1)  #所有类型边下入度之和
            graph.multi_update_all(update_dict, 'sum')  #每种类型边反向聚合消息之和，没有加attention即加权
            bsrc = graph.nodes[src].data['h_b']

            norm_src = torch.pow(in_degrees_b, norm_2)
            shp_src = norm_src.shape + (1,) * (feat_src.dim() - 1)
            norm_src = torch.reshape(norm_src, shp_src)
            bsrc = bsrc * norm_src  #归一化

            return bsrc, rst


def prompt_cat(node_embedding, prompt_embedding, n_node):
    return torch.cat((prompt_embedding.expand(n_node, -1), node_embedding), dim=1)


class GAT(nn.Module):
    def __init__(self, args, graph, in_dim, hidden_dim, out_dim, device):
        super(GAT, self).__init__()
        print(in_dim, hidden_dim, out_dim)
        self.gat = args.gat  #default=0
        self.hid_dim = in_dim
        self.dataset = args.dataset
        self.neg_samples = args.neg_samples  #Number of negative samples
        self.classify_as_edge = args.classify_as_edge  #default=1,only for pretrain
        self.decay = eval(args.regs)[0]  #ser-item loss coeffecient用户-项目损失系数
        self.prompt = args.prompt  #default=0
        n_users = graph.nodes('user').shape[0]
        n_items = graph.nodes('item').shape[0]
        self.user_embedding = torch.nn.Parameter(torch.randn(n_users, self.hid_dim * 2 ** self.prompt)).to(device) #创建一个n_user * (hid_dim*(2^prompt)) 的张量（可学习的）

        self.user_hyperedge = torch.empty((n_users, self.hid_dim * 2 ** (self.prompt)), requires_grad=False).to(device)  #创建一个n_user * (hid_dim*(2^prompt)) 的空张量（不更新梯度的）

        self.item_embedding = torch.nn.Parameter(torch.randn(n_items, self.hid_dim)).to(device) #创建一个n_item * hid_dim 的张量（可学习的）
        self.item_hyperedge = torch.empty((n_items, self.hid_dim * 2 ** (self.prompt)), requires_grad=False).to(device)  #创建一个n_item * (hid_dim*(2^prompt)) 的空张量（不更新梯度的）

        if self.prompt:
            ui_task_embedding = torch.nn.Parameter(torch.randn(1, self.hid_dim)).to(device)  #'ui'='user-item'
            ic_task_embedding = torch.nn.Parameter(torch.randn(1, self.hid_dim)).to(device)  #'ic'='item-cate'
            ir_task_embedding = torch.nn.Parameter(torch.randn(1, self.hid_dim)).to(device)  #'ir'='item-rate'

        self.cate_embedding = torch.nn.Parameter(
            torch.randn(graph.nodes('cate').shape[0], self.hid_dim * 2 ** (self.prompt))).to(device)  #初始化cate的embedding
        self.rate_embedding = torch.nn.Parameter(
            torch.randn(graph.nodes('rate').shape[0], self.hid_dim * 2 ** (self.prompt))).to(device) #初始化rate的embedding
        if 'xmrec' in self.dataset:
            self.bT_embedding = torch.nn.Parameter(
                torch.randn(graph.nodes('bT_idx').shape[0], self.hid_dim * 2 ** (self.prompt))).to(device)
            self.cpr_embedding = torch.nn.Parameter(
                torch.randn(graph.nodes('cpr_idx').shape[0], self.hid_dim * 2 ** (self.prompt))).to(device)
            if self.prompt:
                ib_task_embedding = torch.nn.Parameter(torch.randn(1, self.hid_dim)).to(device)
                ip_task_embedding = torch.nn.Parameter(torch.randn(1, self.hid_dim)).to(device)
        # self.layer1_gu = MultiHeadGATLayer(in_dim, out_dim, num_heads)

        torch.nn.init.normal_(self.user_embedding, std=0.1)  #正态分布初始化，标准差std=0.1
        torch.nn.init.normal_(self.item_embedding, std=0.1)
        self.build_model(device)
        if self.prompt:
            self.node_features = {'user': self.user_embedding,
                                  'item': prompt_cat(self.item_embedding, ui_task_embedding, n_items),
                                  'item_cate': prompt_cat(self.item_embedding, ic_task_embedding, n_items), #item-cate预训练任务中item的embedding与原item的embedding拼接,ic_task_embedding先扩展到n_items行
                                  'item_rate': prompt_cat(self.item_embedding, ir_task_embedding, n_items), #item-rate预训练任务中item的embedding与原item的embedding拼接，ir_task_embedding先扩展到n_items行
                                  'cate': self.cate_embedding, 'rate': self.rate_embedding,
                                  }
            if 'xmrec' in self.dataset:
                self.node_features.update({'bT_idx': self.bT_embedding,
                                           'item_bT_idx': prompt_cat(self.item_embedding, ib_task_embedding, n_items),
                                           'cpr_idx': self.cpr_embedding,
                                           'item_cpr_idx': prompt_cat(self.item_embedding, ip_task_embedding,
                                                                      n_items), })
        else:
            self.node_features = {'user': self.user_embedding,
                                  'item': self.item_embedding,
                                  'cate': self.cate_embedding, 'rate': self.rate_embedding,
                                  }
            if 'xmrec' in self.dataset:
                self.node_features.update({'bT_idx': self.bT_embedding,
                                           'cpr_idx': self.cpr_embedding,
                                           })
        self.pred = ScorePredictor()

    def build_model(self, device):
        self.GATlayers = nn.ModuleList()
        for i in range(10):
            self.GATlayers.append(
                MultiHeadGATLayer(in_dim=self.hid_dim * 2 ** self.prompt, out_dim=self.hid_dim * 2 ** self.prompt,
                                  num_heads=2))  #10个2头的MultiHeadGATLayer

    def forward(self, graph, pre_train=True):
        h = self.node_features
        h_user = self.GATlayers[0](graph, h, ('item', 'iu', 'user'), self.prompt)  #初始item的消息聚合成user的消息h_user
        h_item = self.GATlayers[0](graph, h, ('user', 'ui', 'item'), self.prompt)  #初始user的消息聚合成item的消息h_item
        if pre_train:
            if self.classify_as_edge:
                h_cate_idx = self.GATlayers[0](graph, h, ('item', 'ic', 'cate'), self.prompt) #初始item消息聚合成cate的消息h_cate_idx
                h_item_cate = self.GATlayers[0](graph, h, ('cate', 'ci', 'item'), self.prompt) #初始cate消息聚合成item的消息h_item_cate
                h_rate_idx = self.GATlayers[0](graph, h, ('item', 'ir', 'rate'), self.prompt)
                h_item_rate = self.GATlayers[0](graph, h, ('rate', 'ri', 'item'), self.prompt)
            else:
                h_item_cate = self.GATlayers[0](graph, h, ('cate', 'ci', 'item'), self.prompt) #初始cate消息聚合成item的消息h_item_cate
                h_item_rate = self.GATlayers[0](graph, h, ('rate', 'ri', 'item'), self.prompt) #初始rate消息聚合成item的消息h_item_rate
                h_item_cate = F.softmax(self.cate_fc(h_item_cate), dim=1)  #维度转换后，按行归一化
                h_item_rate = F.softmax(self.rate_fc(h_item_rate), dim=1)
            if 'xmrec' in self.dataset:
                h_item_bT = self.GATlayers[0](graph, h, ('bT_idx', 'bi', 'item'), self.prompt)
                h_bT_idx = self.GATlayers[0](graph, h, ('item', 'ib', 'bT_idx'), self.prompt)
                if self.dataset == 'xmrec_mx':
                    h_item_cpr = self.GATlayers[0](graph, h, ('cpr_idx', 'pi', 'item'), self.prompt)
                    h_cpr_idx = self.GATlayers[0](graph, h, ('item', 'ip', 'cpr_idx'), self.prompt)
                else:
                    h_item_cpr = []
                    h_cpr_idx = []
        else:
            h_item_cate, h_item_rate, h_item_bT, h_item_cpr, h_user_age, h_user_job, h_bT_idx, h_cpr_idx, h_cate_idx, h_rate_idx, h_age_idx, h_job_idx = [], [], [], [], [], [], [], [], [], [], [], []
        if self.classify_as_edge:
            if 'xmrec' in self.dataset:
                h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate, 'item_rate': h_item_rate,
                     'item_bT': h_item_bT, 'item_cpr': h_item_cpr, 'cate': h_cate_idx, 'rate': h_rate_idx,
                     'bT_idx': h_bT_idx, 'cpr_idx': h_cpr_idx}
            else:
                h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate, 'item_rate': h_item_rate,
                     'user_age': h_user_age, 'user_job': h_user_job, 'cate': h_cate_idx, 'rate': h_rate_idx,
                     'age': h_age_idx, 'job': h_job_idx}
        else:
            if 'xmrec' in self.dataset:
                h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate, 'item_rate': h_item_rate,
                     'item_bT': h_item_bT, 'item_cpr': h_item_cpr, 'bT_idx': h_bT_idx, 'cpr_idx': h_cpr_idx}
            else:
                h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate, 'item_rate': h_item_rate,
                     'user_age': h_user_age, 'user_job': h_user_job}
        return h

    def create_bpr_loss(self, pos_g, neg_g, h, user_non_induct=None,loss_type=None):
        sub_fig_feature = {'user': h['user'], 'item': h['item']}
        pos_score = self.pred(pos_g, sub_fig_feature)
        neg_score = self.pred(neg_g, sub_fig_feature)
        pos_score = pos_score[('user', 'ui', 'item')].repeat_interleave(self.neg_samples, dim=0)
        neg_score = neg_score[('user', 'ui', 'item')]
        mf_loss = nn.Softplus()(neg_score - pos_score)
        mf_loss = mf_loss.mean()
        if user_non_induct != None:
            user_embedding = torch.index_select(self.user_embedding, 0, user_non_induct)
            regularizer = (1 / 2) * (user_embedding.norm(2).pow(2) +
                                     self.item_embedding.norm(2).pow(2))
        else:
            regularizer = (1 / 2) * (self.user_embedding.norm(2).pow(2) +
                                     self.item_embedding.norm(2).pow(2))
        emb_loss = self.decay * regularizer

        bpr_loss = mf_loss + emb_loss
        return bpr_loss, mf_loss, emb_loss

    def create_item_bpr_loss(self, pos_g, neg_g, h, node_type='bT_idx', n_type=12):
        edge_dict = {'bT_idx': 'bi', 'cpr_idx': 'pi', 'cate': 'ci', 'rate': 'ri'}
        item_dict = {'bT_idx': 'item_bT', 'cpr_idx': 'item_cpr', 'cate': 'item_cate', 'rate': 'item_rate'}
        repeat_dict = {'bT_idx': self.neg_samples, 'cpr_idx': self.neg_samples, 'cate': n_type - 1, 'rate': n_type - 1}
        sub_fig_feature = {node_type: h[node_type], 'item': h[item_dict[node_type]]}
        pos_score = self.pred(pos_g, sub_fig_feature)
        neg_score = self.pred(neg_g, sub_fig_feature)
        # pos_score = pos_score[(node_type, edge_dict[node_type], 'item')].repeat_interleave(self.neg_samples, dim=0)
        pos_score = pos_score[(node_type, edge_dict[node_type], 'item')].repeat_interleave(repeat_dict[node_type],
                                                                                           dim=0)
        neg_score = neg_score[(node_type, edge_dict[node_type], 'item')]
        mf_loss = nn.Softplus()(neg_score - pos_score)
        mf_loss = mf_loss.mean()
        regularizer = (1 / 2) * (h[node_type].norm(2).pow(2) +
                                 h[item_dict[node_type]].norm(2).pow(2))
        emb_loss = self.decay * regularizer

        bpr_loss = mf_loss + emb_loss
        return bpr_loss, mf_loss, emb_loss

    def create_classify_loss(self, item_embedding, label):
        return F.cross_entropy(item_embedding, label, reduction='mean')


class LightGCN(nn.Module):
    def __init__(self, args, graph, device):
        super().__init__()
        self.hid_dim = args.embed_size  #default=128
        self.layer_num = args.layer_num #Output sizes of every layer, default=3
        self.device = device
        self.n_cate = graph.nodes('cate').shape[0]
        self.n_rate = graph.nodes('rate').shape[0]

        self.classify_as_edge = args.classify_as_edge
        self.neg_samples = args.neg_samples  #负样本数目
        self.decay = eval(args.regs)[1]  #嵌入正则化
        self.ui_loss_alpha = eval(args.regs)[0] #用户项目损失系数
        self.dataset = args.dataset
        self.pre_train = args.pre_train  #default=1
        self.n_user = graph.nodes('user').shape[0]
        self.lightgcn = args.lightgcn
        self.pre_gcn = args.pre_gcn  #defalut=0
        self.trans_alpha = eval(args.hgcn_mix)[0]  #default=10
        self.resid_beta = eval(args.hgcn_mix)[1]  #default=0.1
        self.hgcn = args.hgcn  #default=1
        self.norm_2 = args.norm_2  #default=-1,-0.5 for mx, -1 for cn/mx_C2EP
        self.att_conv = args.att_conv  #-1 if no trans_conv
        self.contrastive_learning = args.contrastive_learning  #default=0

        self.user_embedding = torch.nn.Parameter(torch.randn(graph.nodes('user').shape[0], self.hid_dim))  #初始化用户embedding
        self.user_hyperedge = torch.empty((graph.nodes('user').shape[0], self.hid_dim), requires_grad=False)  #创建空的用户embedding，存放的是由超边聚合得到的用户embedding

        self.item_embedding = torch.nn.Parameter(torch.randn(graph.nodes('item').shape[0], self.hid_dim))  #初始化项目embedding
        self.item_hyperedge = torch.empty((graph.nodes('item').shape[0], self.hid_dim), requires_grad=False)  #创建空的项目embedding，存放是的由超边聚合得到的项目embedding

        self.cate_hyperedge = torch.empty((graph.nodes('cate').shape[0], self.hid_dim), requires_grad=False)  #创建空的cate节点embedding
        self.rate_hyperedge = torch.empty((graph.nodes('rate').shape[0], self.hid_dim), requires_grad=False)  #创建空的rate节点embedding
        if 'xmrec' in self.dataset:
            self.bT_hyperedge = torch.empty((graph.nodes('bT_idx').shape[0], self.hid_dim), requires_grad=False)
            self.cpr_hyperedge = torch.empty((graph.nodes('cpr_idx').shape[0], self.hid_dim), requires_grad=False)
        else:
            self.age_hyperedge = torch.empty((graph.nodes('age').shape[0], self.hid_dim), requires_grad=False)
            self.job_hyperedge = torch.empty((graph.nodes('job').shape[0], self.hid_dim), requires_grad=False)  #根据不同数据集，用户有不同的预训练超图，及对应的超边节点embedding

        torch.nn.init.normal_(self.user_embedding, std=0.1)
        torch.nn.init.normal_(self.item_embedding, std=0.1)  #正态分布，只有这俩是随梯度更新的

        self.dropout = nn.Dropout(p=0.2)  #指定一个dropout层，训练时丢弃20%的输入张量
        self.edge_dropout = dgl.transforms.DropEdge(p=0.2)  #丢弃图中的边
        self.build_model()
        self.node_features = {'user': self.user_embedding, 'item': self.item_embedding,
                              'user_edge': self.user_hyperedge, 'item_edge': self.item_hyperedge,
                              'cate_edge': self.cate_hyperedge, 'rate_edge': self.rate_hyperedge,
                              }
        if 'xmrec' in self.dataset:
            self.node_features.update({'bT_idx_edge': self.bT_hyperedge, 'cpr_idx_edge': self.cpr_hyperedge})
        else:
            self.node_features.update({'age_edge': self.age_hyperedge, 'job_edge': self.job_hyperedge})
        self.pred = ScorePredictor()


    def build_layer(self, idx=0):
        return LightGCNLayer()

    def load_pretrain_embedding(self):
        print('load pretrained embedding...')
        pretrained_user_embedding = np.load('./weights/' + self.dataset + '/user.npy')
        pretrained_user_embedding = torch.from_numpy(pretrained_user_embedding)
        print(pretrained_user_embedding.shape)
        # sys.exit(0)
        return pretrained_user_embedding

    def build_model(self):
        self.HGCNlayer = HGCNLayer()
        self.HGCNlayer_general = HGCNLayer_general()
        self.layers = nn.ModuleList()
        self.LGCNlayer = LightGCNLayer()
        self.cate_fc = nn.Linear(self.hid_dim, self.n_cate)
        self.rate_fc = nn.Linear(self.hid_dim, self.n_rate)
        for idx in range(self.layer_num):
            h2h = self.build_layer(idx)
            self.layers.append(h2h)

    def MF_forward(self):
        h = self.node_features
        return h

    def lightgcn_forward(self, graph):
        h = self.node_features
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]
        for layer in self.layers:
            h_item = layer(graph, h, ('user', 'ui', 'item'))
            h_user = layer(graph, h, ('item', 'iu', 'user'))
            h = {'user': h_user, 'item': h_item}
            user_embed.append(h_user)
            item_embed.append(h_item)
        user_embed = torch.mean(torch.stack(user_embed, dim=0), dim=0)
        item_embed = torch.mean(torch.stack(item_embed, dim=0), dim=0)
        h = {'user': user_embed, 'item': item_embed}
        return h

    def forward(self, graph, pre_train=True):
        h = self.node_features
        norm = self.norm_2
        if self.hgcn:
            # if pre_train:
            #     user_nodes = graph.nodes('user')
            #     mask = torch.bernoulli(torch.ones_like(user_nodes) * 0.001) > 0
            #     masked_user = torch.masked_select(user_nodes, mask)
            #     ui_subgraph = dgl.node_subgraph(graph, {'user': graph.nodes('user'), 'item': graph.nodes('item')})
            #     ui_subgraph = dgl.remove_edges(ui_subgraph, masked_user, etype='ui')
            #     ui_subgraph = dgl.remove_edges(ui_subgraph, masked_user, etype='iu')
            # else:
            #     ui_subgraph = graph

            #没有pretrain信息融入的HGCNlayer要更改，加入池化和反池化
            h_user, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), norm)
            h_item, _ = self.HGCNlayer(graph, h, ('item', 'iu', 'user'), ('user', 'ui', 'item'), norm)
            if pre_train:
                if self.classify_as_edge:
                    h_item_cate, h_cate_idx = self.HGCNlayer(graph, h, ('item', 'ic', 'cate'), ('cate', 'ci', 'item'),
                                                             norm)
                    h_item_rate, h_rate_idx = self.HGCNlayer(graph, h, ('item', 'ir', 'rate'), ('rate', 'ri', 'item'),
                                                             norm)


                else:
                    h_item_cate, _ = self.HGCNlayer(graph, h, ('item', 'ic', 'cate'), ('cate', 'ci', 'item'), norm)
                    h_item_rate, _ = self.HGCNlayer(graph, h, ('item', 'ir', 'rate'), ('rate', 'ri', 'item'), norm)
                    h_item_cate = F.softmax(self.cate_fc(h_item_cate), dim=1)
                    h_item_rate = F.softmax(self.rate_fc(h_item_rate), dim=1)
                    # if 'steam' in self.dataset:
                    #     h_user_age, _ = self.HGCNlayer(graph, h, ('user', 'ua', 'age'), ('age', 'au', 'user'), norm)
                    #     h_user_job, _ = self.HGCNlayer(graph, h, ('user', 'uj', 'job'), ('job', 'ju', 'user'), norm)
                    #     h_user_age = F.softmax(self.cate_fc(h_user_age), dim=1)
                    #     h_user_job = F.softmax(self.rate_fc(h_user_job), dim=1)
                if 'xmrec' in self.dataset:
                    h_item_bT, h_bT_idx = self.HGCNlayer(graph, h, ('item', 'ib', 'bT_idx'), ('bT_idx', 'bi', 'item'),
                                                         norm)
                    h_item_cpr, h_cpr_idx = self.HGCNlayer(graph, h, ('item', 'ip', 'cpr_idx'),
                                                           ('cpr_idx', 'pi', 'item'),
                                                           norm)
                if 'steam' in self.dataset:
                    h_user_age, h_age_idx = self.HGCNlayer(graph, h, ('user', 'ua', 'age'), ('age', 'au', 'user'),
                                                           norm)
                    h_user_job, h_job_idx = self.HGCNlayer(graph, h, ('user', 'uj', 'job'), ('job', 'ju', 'user'),
                                                           norm)
            else:
                h_item_cate, h_item_rate, h_item_bT, h_item_cpr, h_user_age, h_user_job, h_bT_idx, h_cpr_idx, h_cate_idx, h_rate_idx, h_age_idx, h_job_idx = [], [], [], [], [], [], [], [], [], [], [], []
            if self.classify_as_edge:
                if 'xmrec' in self.dataset:
                    h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate, 'item_rate': h_item_rate,
                         'item_bT': h_item_bT, 'item_cpr': h_item_cpr, 'cate': h_cate_idx, 'rate': h_rate_idx,
                         'bT_idx': h_bT_idx, 'cpr_idx': h_cpr_idx}
                else:
                    h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate, 'item_rate': h_item_rate,
                         'user_age': h_user_age, 'user_job': h_user_job, 'cate': h_cate_idx, 'rate': h_rate_idx,
                         'age': h_age_idx, 'job': h_job_idx}
            else:
                if 'xmrec' in self.dataset:
                    h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate, 'item_rate': h_item_rate,
                         'item_bT': h_item_bT, 'item_cpr': h_item_cpr, 'bT_idx': h_bT_idx, 'cpr_idx': h_cpr_idx}
                else:
                    h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate, 'item_rate': h_item_rate,
                         'user_age': h_user_age, 'user_job': h_user_job, 'age': h_age_idx, 'job': h_job_idx}
        '''Transitional HyperConv'''
        if pre_train and self.att_conv!=-1:
            '''TODO: xmrec_cn and steam - finetune transition strength'''
            h['item_edge'], h['user_edge'] = self.item_hyperedge, self.user_hyperedge
            if 'xmrec' in self.dataset:
                if self.att_conv == 0:
                    h_user_v2, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), -1,
                                                  self.trans_alpha,
                                                  h_item_cate + h_item_rate + h_item_bT + h_item_cpr, detach=True)
                elif self.att_conv == 2:
                    h_user_v2, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), -1,
                                                  self.trans_alpha, torch.cat((h_item_cate, h_item_rate,h_item_bT,h_item_cpr), dim=1),
                                                  detach=True,att=self.att_conv)
                else:
                    h_user_v2, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), -1,
                                                  self.trans_alpha,
                                                  torch.cat(
                                                      (torch.unsqueeze(h_item_cate, 1), torch.unsqueeze(h_item_rate, 1),
                                                       torch.unsqueeze(h_item_bT, 1), torch.unsqueeze(h_item_cpr, 1)
                                                       ),
                                                      dim=1), detach=True, att=self.att_conv)  #经过torch.unsqueeze后n*1*d矩阵，再cat后，是n*4*d矩阵
            elif 'steam' in self.dataset:
                if self.att_conv == 0:
                    h_user_v2, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), -1,
                                                  self.trans_alpha, h_item_cate + h_item_rate, detach=True)
                    h_item_v2, _ = self.HGCNlayer(graph, h, ('item', 'iu', 'user'), ('user', 'ui', 'item'), -1,
                                                  self.trans_alpha, h_user_age + h_user_job, detach=True)
                elif self.att_conv == 2:
                    h_user_v2, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), -1,
                                                  self.trans_alpha, torch.cat((h_item_cate,h_item_rate),dim=1), detach=True,att=self.att_conv)
                    h_item_v2, _ = self.HGCNlayer(graph, h, ('item', 'iu', 'user'), ('user', 'ui', 'item'), -1,
                                                  self.trans_alpha, torch.cat((h_user_age,h_user_job),dim=1), detach=True,att=self.att_conv)

                else:
                    h_user_v2, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), -1,
                                                  self.trans_alpha,
                                                  torch.cat(
                                                      (
                                                      torch.unsqueeze(h_item_cate, 1), torch.unsqueeze(h_item_rate, 1)),
                                                      dim=1), detach=True, att=self.att_conv)
                    h_item_v2, _ = self.HGCNlayer(graph, h, ('item', 'iu', 'user'), ('user', 'ui', 'item'), -1,
                                                  self.trans_alpha,
                                                  torch.cat(
                                                      (torch.unsqueeze(h_user_age, 1), torch.unsqueeze(h_user_job, 1)),
                                                      dim=1), detach=True, att=self.att_conv)
            if 'xmrec' in self.dataset:
                h_user = h_user + self.resid_beta * h_user_v2
                h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate, 'item_rate': h_item_rate,
                     'item_bT': h_item_bT, 'item_cpr': h_item_cpr, 'cate': h_cate_idx, 'rate': h_rate_idx,
                     'bT_idx': h_bT_idx, 'cpr_idx': h_cpr_idx}
            else:
                h_user = h_user + self.resid_beta * h_user_v2
                h_item = h_item + self.resid_beta * h_item_v2
                h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate,
                     'item_rate': h_item_rate,
                     'user_age': h_user_age, 'user_job': h_user_job, 'cate': h_cate_idx, 'rate': h_rate_idx,
                     'age': h_age_idx, 'job': h_job_idx}
        return h

    # def alignment_score(self):
    #     h_user_v1 = F.normalize(self.h_user_v1, p=2, dim=-1)
    #     h_user_v2 = F.normalize(self.h_user_v2, p=2, dim=-1)
    #     cosine_similarity = torchmetrics.CosineSimilarity(reduction=None)
    #     result = cosine_similarity(h_user_v1, h_user_v2).cpu().detach().numpy()
    #     return np.nanmean(result)
    #
    # def uniformity_score_user(self):
    #     h_user_v1 = F.normalize(self.h_user_v1, p=2, dim=-1)
    #     result = torchmetrics.functional.pairwise_cosine_similarity(h_user_v1).cpu().detach().numpy()
    #     return np.nanmean(result)
    #
    # def uniformity_score_item(self):
    #     h_item_v1 = F.normalize(self.h_item_v1, p=2, dim=-1)
    #     return torchmetrics.functional.pairwise_cosine_similarity(h_item_v1).mean()
    #
    # def uniformity_score_item_batch(self):
    #     output = 0
    #     k = 4
    #     h_item_v1 = np.array_split(F.normalize(self.h_item_v1, p=2, dim=-1).cpu().detach().numpy(), k)
    #     for idx in range(k):
    #         output += sklearn.metrics.pairwise.cosine_similarity(h_item_v1[idx]).mean()
    #     return output / k

    def create_ssl_loss_user(self, ssl_temp):
        # ssl_temp = 0.1
        h_user_v1 = torch.nn.functional.normalize(self.h_user_v1, p=2, dim=1)  #使用L2范数归一化
        h_user_v2 = torch.nn.functional.normalize(self.h_user_v2, p=2, dim=1)
        pos_score = torch.sum(torch.mul(h_user_v1, h_user_v2), dim=1)  #对应元素相乘后按行求和，pos_score[i]表示user_i消息v1和消息v2的内积
        neg_score = torch.matmul(h_user_v1, h_user_v2.T)  #neg_score[i,j]表示user_i的消息v1和user_j的消息v2的的内积，n*n
        pos_score = torch.exp(pos_score / ssl_temp)
        neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)  #neg_score[i]表示user_i的消息v1与所有user的消息v2内积之和
        ssl_loss = -torch.sum(torch.log(pos_score / neg_score))
        return ssl_loss

    def create_ssl_loss_item(self, ssl_temp):
        ssl_temp = 0.1
        h_item_v1 = torch.nn.functional.normalize(self.h_item_v1, p=2, dim=1)
        h_item_v2 = torch.nn.functional.normalize(self.h_item_v2, p=2, dim=1)
        # pos_score = torch.sum(torch.mul(h_item_v1, h_item_v2), dim=1)
        neg_score = torch.matmul(h_item_v1, h_item_v2.T)
        pos_score = 1 / ssl_temp
        neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)
        ssl_loss = -torch.sum(torch.log(pos_score / neg_score))
        return ssl_loss

    def create_ssl_loss_batched_item(self, ssl_temp, k=4, idx=0):
        ssl_temp = 0.1
        h_item_v1 = self.h_item_v1.split(self.h_item_v1.shape[0] // k + 1)[idx]  #分为k+1个batch，这为第idx个batch
        h_item_v2 = self.h_item_v2.split(self.h_item_v2.shape[0] // k + 1)[idx]
        h_item_v1 = torch.nn.functional.normalize(h_item_v1, p=2, dim=1)
        h_item_v2 = torch.nn.functional.normalize(h_item_v2, p=2, dim=1)
        neg_score = torch.matmul(h_item_v1, h_item_v2.T)
        pos_score = 1 / ssl_temp
        neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)
        ssl_loss = -torch.sum(torch.log(pos_score / neg_score))
        return ssl_loss

    def reweight_formula(self, beta, x, type=0):
        if type == 0:
            fx = (1 - beta) / (1 - beta ** x)
        elif type == 1:
            fx = 1 / (beta * x - beta + 1)
        elif type == 2:
            fx = 1 / torch.exp(beta * x - beta)
        elif type == 3:
            fx = 1 - torch.tanh(beta * x - beta)
        return torch.unsqueeze(fx, 1)

    def uniformity_loss(self, h_user):  #均匀性损失
        x = h_user[torch.randint(len(h_user), (2000,))]  #随机生成2000个索引，选择h_user中对应元素
        x = F.normalize(x, dim=-1)  #沿着最后一个维度归一化
        result = torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()  #torch.pdist(x,p=2)计算张量x中两两样本的欧几里得距离
        # print(result)
        # sys.exit()
        return 0.005 * result

    def create_bpr_loss(self, pos_g, neg_g, h, users_non_induct=None, loss_type='bpr'):
        if loss_type == 'align':
            # h['user'] = F.normalize(h['user'], dim=-1)
            # h['item'] = F.normalize(h['item'], dim=-1)
            sub_fig_feature = {'user': h['user'], 'item': h['item']}
            pos_score = self.pred.alignment_forward(pos_g, sub_fig_feature)  #边两端节点差值表示边对齐score
            mf_loss = pos_score[('user', 'ui', 'item')].norm(p=2, dim=1).pow(2).mean()  #按行归一化，平方，求平均
            # mf_loss = nn.Softplus()(pos_score[('user', 'ui', 'item')]).mean()
        elif loss_type == 'infonce':
            ssl_temp = 1e9
            # sub_fig_feature = {'user': F.normalize(h['user'],p=2, dim=1), 'item': F.normalize(h['item'],p=2, dim=1)}
            sub_fig_feature = {'user': h['user'], 'item': h['item']}
            pos_score = self.pred(pos_g, sub_fig_feature)
            pos_score = torch.exp(pos_score[('user', 'ui', 'item')] / ssl_temp)
            neg_score = self.pred(neg_g, sub_fig_feature)
            neg_score = neg_score[('user', 'ui', 'item')]
            neg_score = neg_score.reshape((pos_score.shape[0], self.neg_samples))
            neg_score = torch.exp(neg_score.sum(dim=1).unsqueeze(1) / ssl_temp)
            ssl_loss = -torch.sum(torch.log(pos_score / neg_score))
            mf_loss = ssl_loss
        elif loss_type == 'infoa':
            ssl_temp = 1e4
            # sub_fig_feature = {'user': F.normalize(h['user'],p=2, dim=1), 'item': F.normalize(h['item'],p=2, dim=1)}
            sub_fig_feature = {'user': h['user'], 'item': h['item']}
            pos_score = self.pred.alignment_forward(pos_g, sub_fig_feature)
            pos_score = torch.exp(pos_score[('user', 'ui', 'item')].norm(p=2, dim=1).pow(2) / ssl_temp)
            neg_score = self.pred.alignment_forward(neg_g, sub_fig_feature)
            neg_score = neg_score[('user', 'ui', 'item')].norm(p=2, dim=1).pow(2)
            neg_score = neg_score.reshape((pos_score.shape[0], self.neg_samples))
            neg_score = torch.exp(neg_score.sum(dim=1) / ssl_temp)
            ssl_loss = -torch.mean(torch.log(neg_score / pos_score))
            # ssl_loss = nn.Softplus()(pos_score - neg_score)
            mf_loss = ssl_loss.mean()
        elif loss_type == 'bpra':
            ssl_temp = 1e4
            # sub_fig_feature = {'user': F.normalize(h['user'],p=2, dim=1), 'item': F.normalize(h['item'],p=2, dim=1)}
            sub_fig_feature = {'user': h['user'], 'item': h['item']}
            pos_score = self.pred.alignment_forward(pos_g, sub_fig_feature)
            pos_score = pos_score[('user', 'ui', 'item')].norm(p=2, dim=1).pow(2).mean()
            neg_score = self.pred.alignment_forward(neg_g, sub_fig_feature)
            neg_score = neg_score[('user', 'ui', 'item')].norm(p=2, dim=1).pow(2).mean()
            ssl_loss = nn.Softplus()(pos_score - neg_score)
            mf_loss = ssl_loss
        elif loss_type == 'bpr_pos':
            sub_fig_feature = {'user': h['user'], 'item': h['item']}
            pos_score = self.pred(pos_g, sub_fig_feature)
            pos_score = pos_score[('user', 'ui', 'item')]
            mf_loss = nn.Softplus()(-pos_score)
            mf_loss = mf_loss.mean()
        else:
            sub_fig_feature = {'user': h['user'], 'item': h['item']}
            pos_score = self.pred(pos_g, sub_fig_feature)
            neg_score = self.pred(neg_g, sub_fig_feature)
            pos_score = pos_score[('user', 'ui', 'item')].repeat_interleave(self.neg_samples, dim=0)
            neg_score = neg_score[('user', 'ui', 'item')]
            mf_loss = nn.Softplus()(neg_score - pos_score)
            mf_loss = mf_loss.mean()
        if users_non_induct != None:
            regularizer = (1 / 2) * (self.user_embedding[users_non_induct].norm(2).pow(2) +
                                     self.item_embedding.norm(2).pow(2))
        else:
            regularizer = (1 / 2) * (self.user_embedding.norm(2).pow(2) +
                                     self.item_embedding.norm(2).pow(2))  #正则化项
        emb_loss = self.decay * regularizer

        # bpr_loss = self.ui_loss_alpha * mf_loss + emb_loss
        bpr_loss = mf_loss + emb_loss
        return bpr_loss, mf_loss, emb_loss

    '''TODO: bpr_loss for boughtTogether and ComparedTogether items link-prediction task'''

    def alignment(self, x, y):  #求x,y的对齐损失
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):  #求x的均匀性损失
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def create_ssl_loss_batched_group(self, ssl_temp, k=4, idx=0):
        ssl_temp = 0.1
        h_group_v1 = self.h_group_v1.split(self.h_group_v1.shape[0] // k + 1)[idx]
        h_group_v2 = self.h_group_v2.split(self.h_group_v2.shape[0] // k + 1)[idx]
        h_group_v1 = torch.nn.functional.normalize(h_group_v1, p=2, dim=1)
        h_group_v2 = torch.nn.functional.normalize(h_group_v2, p=2, dim=1)
        # pos_score = torch.sum(torch.mul(h_group_v1, h_group_v2), dim=1)
        neg_score = torch.matmul(h_group_v1, h_group_v2.T)
        pos_score = 1 / ssl_temp
        neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)
        ssl_loss = -torch.sum(torch.log(pos_score / neg_score))
        return ssl_loss

    def uniformity_loss_batched(self, x, k=4, idx=0):
        x = x.split(x.shape[0] // k + 1)[idx]
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_AU_loss(self, pos_g, h, k=4, idx=0):
        user_e, item_e = h['user'], h['item']
        user_e_batched = user_e.split(user_e.shape[0] // k + 1)[idx]
        item_e_batched = item_e.split(item_e.shape[0] // k + 1)[idx]
        related_item_e = item_e[pos_g.edges(etype='ui')[1]]
        related_item_e_batched = related_item_e.split(related_item_e.shape[0] // k + 1)[idx]
        align = self.alignment(user_e_batched, related_item_e_batched)
        uniform = (self.uniformity(user_e_batched) + self.uniformity(item_e_batched)) / 2
        loss = align + 1 * uniform
        return loss

    def create_item_bpr_loss(self, pos_g, neg_g, h, node_type='bT_idx', n_type=12, loss_type='bpr'):
        edge_dict = {'bT_idx': 'bi', 'cpr_idx': 'pi', 'cate': 'ci', 'rate': 'ri'}
        item_dict = {'bT_idx': 'item_bT', 'cpr_idx': 'item_cpr', 'cate': 'item_cate', 'rate': 'item_rate'}
        repeat_dict = {'bT_idx': self.neg_samples, 'cpr_idx': self.neg_samples, 'cate': n_type - 1, 'rate': n_type - 1}
        if loss_type == 'align':
            sub_fig_feature = {node_type: h[node_type], 'item': h[item_dict[node_type]]}
            pos_score = self.pred.alignment_forward(pos_g, sub_fig_feature)
            mf_loss = pos_score[(node_type, edge_dict[node_type], 'item')].norm(p=2, dim=1).pow(2).mean()
        else:
            sub_fig_feature = {node_type: h[node_type], 'item': h[item_dict[node_type]]}
            pos_score = self.pred(pos_g, sub_fig_feature)
            neg_score = self.pred(neg_g, sub_fig_feature)
            # pos_score = pos_score[(node_type, edge_dict[node_type], 'item')].repeat_interleave(self.neg_samples, dim=0)
            pos_score = pos_score[(node_type, edge_dict[node_type], 'item')].repeat_interleave(repeat_dict[node_type],
                                                                                               dim=0)
            neg_score = neg_score[(node_type, edge_dict[node_type], 'item')]
            mf_loss = nn.Softplus()(neg_score - pos_score)
            mf_loss = mf_loss.mean()
        regularizer = (1 / 2) * (h[node_type].norm(2).pow(2) +
                                 h[item_dict[node_type]].norm(2).pow(2))
        emb_loss = self.decay * regularizer

        bpr_loss = mf_loss + emb_loss
        return bpr_loss, mf_loss, emb_loss

    def create_user_bpr_loss(self, pos_g, neg_g, h, node_type='age', loss_type='bpr'):
        edge_dict = {'age': 'au', 'job': 'ju'}
        item_dict = {'age': 'user_age', 'job': 'user_job'}
        repeat_dict = {'age': self.neg_samples, 'job': self.neg_samples}
        if loss_type == 'align':
            sub_fig_feature = {node_type: h[node_type], 'user': h[item_dict[node_type]]}
            pos_score = self.pred.alignment_forward(pos_g, sub_fig_feature)
            mf_loss = pos_score[(node_type, edge_dict[node_type], 'user')].norm(p=2, dim=1).pow(2).mean()
        else:
            sub_fig_feature = {node_type: h[node_type], 'user': h[item_dict[node_type]]}
            # print(pos_g)
            pos_score = self.pred(pos_g, sub_fig_feature)
            neg_score = self.pred(neg_g, sub_fig_feature)
            # pos_score = pos_score[(node_type, edge_dict[node_type], 'item')].repeat_interleave(self.neg_samples, dim=0)
            pos_score = pos_score[(node_type, edge_dict[node_type], 'user')].repeat_interleave(repeat_dict[node_type],
                                                                                               dim=0)
            neg_score = neg_score[(node_type, edge_dict[node_type], 'user')]
            mf_loss = nn.Softplus()(neg_score - pos_score)
            mf_loss = mf_loss.mean()
        regularizer = (1 / 2) * (h[node_type].norm(2).pow(2) +
                                 h[item_dict[node_type]].norm(2).pow(2))
        emb_loss = self.decay * regularizer

        bpr_loss = mf_loss + emb_loss
        return bpr_loss, mf_loss, emb_loss

    def create_user_bpr_loss_v2(self, pos_g, neg_g, h, node_type='age', n_type=7):
        edge_dict = {'age': 'au', 'job': 'ju'}
        user_dict = {'age': 'user_age', 'job': 'user_job'}
        repeat_dict = {'age': n_type - 1, 'job': n_type - 1}
        sub_fig_feature = {node_type: h[node_type], 'user': h[user_dict[node_type]]}
        pos_score = self.pred(pos_g, sub_fig_feature)
        neg_score = self.pred(neg_g, sub_fig_feature)
        pos_score = pos_score[(node_type, edge_dict[node_type], 'user')].repeat_interleave(repeat_dict[node_type],
                                                                                           dim=0)
        neg_score = neg_score[(node_type, edge_dict[node_type], 'user')]
        mf_loss = nn.Softplus()(neg_score - pos_score)
        mf_loss = mf_loss.mean()
        regularizer = (1 / 2) * (h[node_type].norm(2).pow(2) +
                                 h[user_dict[node_type]].norm(2).pow(2))
        emb_loss = self.decay * regularizer

        bpr_loss = mf_loss + emb_loss
        return bpr_loss, mf_loss, emb_loss

    def create_classify_loss(self, item_embedding, label): #j交叉熵损失，衡量模型输出与真实标签之间差异
        return F.cross_entropy(item_embedding, label, reduction='mean')


class LightGCNP(nn.Module):
    def __init__(self, args, graph, device):
        super().__init__()
        self.pool_beta = args.beta_pool
        self.hid_dim = args.embed_size  #default=128
        self.layer_num = args.layer_num #Output sizes of every layer, default=3
        self.device = device
        self.n_cate = graph.nodes('cate').shape[0]
        self.n_rate = graph.nodes('rate').shape[0]

        self.classify_as_edge = args.classify_as_edge
        self.neg_samples = args.neg_samples  #负样本数目
        self.decay = eval(args.regs)[1]  #嵌入正则化 1e-4
        self.ui_loss_alpha = eval(args.regs)[0] #用户项目损失系数 0.8
        self.dataset = args.dataset
        self.pre_train = args.pre_train  #default=1
        self.n_user = graph.nodes('user').shape[0]
        self.lightgcn = args.lightgcn  #baseline???
        self.pre_gcn = args.pre_gcn  #defalut=0
        self.trans_alpha = eval(args.hgcn_mix)[0]  #default=10
        self.resid_beta = eval(args.hgcn_mix)[1]  #default=0.1
        self.hgcn = args.hgcn  #default=1
        self.norm_2 = args.norm_2  #default=-1,-0.5 for mx, -1 for cn/mx_C2EP
        self.att_conv = args.att_conv  #-1 if no trans_conv
        self.contrastive_learning = args.contrastive_learning  #default=0

        self.user_embedding = torch.nn.Parameter(torch.randn(graph.nodes('user').shape[0], self.hid_dim))  #初始化用户embedding
        self.user_hyperedge = torch.empty((graph.nodes('user').shape[0], self.hid_dim), requires_grad=False)  #创建空的用户embedding，存放的是由超边聚合得到的用户embedding

        self.item_embedding = torch.nn.Parameter(torch.randn(graph.nodes('item').shape[0], self.hid_dim))  #初始化项目embedding
        self.item_hyperedge = torch.empty((graph.nodes('item').shape[0], self.hid_dim), requires_grad=False)  #创建空的项目embedding，存放是的由超边聚合得到的项目embedding

        self.cate_hyperedge = torch.empty((graph.nodes('cate').shape[0], self.hid_dim), requires_grad=False)  #创建空的cate节点embedding
        self.rate_hyperedge = torch.empty((graph.nodes('rate').shape[0], self.hid_dim), requires_grad=False)  #创建空的rate节点embedding
        if 'xmrec' in self.dataset:
            self.bT_hyperedge = torch.empty((graph.nodes('bT_idx').shape[0], self.hid_dim), requires_grad=False)
            self.cpr_hyperedge = torch.empty((graph.nodes('cpr_idx').shape[0], self.hid_dim), requires_grad=False)
        else:
            self.age_hyperedge = torch.empty((graph.nodes('age').shape[0], self.hid_dim), requires_grad=False)
            self.job_hyperedge = torch.empty((graph.nodes('job').shape[0], self.hid_dim), requires_grad=False)  #根据不同数据集，用户有不同的预训练超图，及对应的超边节点embedding

        torch.nn.init.normal_(self.user_embedding, std=0.1)
        torch.nn.init.normal_(self.item_embedding, std=0.1)  #正态分布，只有这俩是随梯度更新的torch.nn.Parameter

        self.dropout = nn.Dropout(p=0.2)  #指定一个dropout层，训练时丢弃20%的输入张量
        self.edge_dropout = dgl.transforms.DropEdge(p=0.2)  #丢弃图中的边
        self.build_model()
        self.node_features = {'user': self.user_embedding, 'item': self.item_embedding,
                              'user_edge': self.user_hyperedge, 'item_edge': self.item_hyperedge,
                              'cate_edge': self.cate_hyperedge, 'rate_edge': self.rate_hyperedge,
                              }
        if 'xmrec' in self.dataset:
            self.node_features.update({'bT_idx_edge': self.bT_hyperedge, 'cpr_idx_edge': self.cpr_hyperedge})
        else:
            self.node_features.update({'age_edge': self.age_hyperedge, 'job_edge': self.job_hyperedge})
        self.pred = ScorePredictor()


    def build_layer(self, idx=0):
        return LightGCNLayer()

    def load_pretrain_embedding(self):
        print('load pretrained embedding...')
        pretrained_user_embedding = np.load('./weights/' + self.dataset + '/user.npy')
        pretrained_user_embedding = torch.from_numpy(pretrained_user_embedding)
        print(pretrained_user_embedding.shape)
        # sys.exit(0)
        return pretrained_user_embedding

    def build_model(self):
        self.HGCNlayer = HGCNLayer()
        self.SEP_U = SEP_U(embed_size=self.hid_dim, beta=self.pool_beta , device=self.device)  #一个SEP_U还是多个SEP_U？？？
        self.HGCNlayer_general = HGCNLayer_general()
        self.layers = nn.ModuleList()
        self.LGCNlayer = LightGCNLayer()
        self.cate_fc = nn.Linear(self.hid_dim, self.n_cate)
        self.rate_fc = nn.Linear(self.hid_dim, self.n_rate)
        for idx in range(self.layer_num):
            h2h = self.build_layer(idx)
            self.layers.append(h2h)

    def MF_forward(self):
        h = self.node_features
        return h

    def lightgcn_forward(self, graph):
        h = self.node_features
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]
        for layer in self.layers:
            h_item = layer(graph, h, ('user', 'ui', 'item'))
            h_user = layer(graph, h, ('item', 'iu', 'user'))
            h = {'user': h_user, 'item': h_item}
            user_embed.append(h_user)
            item_embed.append(h_item)
        user_embed = torch.mean(torch.stack(user_embed, dim=0), dim=0)
        item_embed = torch.mean(torch.stack(item_embed, dim=0), dim=0)
        h = {'user': user_embed, 'item': item_embed}
        return h

    def forward(self, graph, partition, pre_train=True):
        h = self.node_features
        norm = self.norm_2
        if self.hgcn:
            #没有pretrain信息融入的HGCNlayer变为SEP_U(graph, h0, etype_forward, etype_back, norm_2)
            #h_user, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), norm)
            h_user, _ = self.SEP_U(partition['ui'], graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), norm)
            h_item, _ = self.HGCNlayer(graph, h, ('item', 'iu', 'user'), ('user', 'ui', 'item'), norm)
            #h_item, _ = self.SEP_U(partition['i'], graph, h, ('item', 'iu', 'user'), ('user', 'ui', 'item'), norm)

            if pre_train:
                if self.classify_as_edge:
                    h_item_cate, h_cate_idx = self.SEP_U(partition['ic'], graph, h, ('item', 'ic', 'cate'), ('cate', 'ci', 'item'),
                                                             norm)
                    h_item_rate, h_rate_idx = self.SEP_U(partition['ir'], graph, h, ('item', 'ir', 'rate'), ('rate', 'ri', 'item'),
                                                             norm)
                else:
                    h_item_cate, _ = self.SEP_U(partition['ic'], graph, h, ('item', 'ic', 'cate'), ('cate', 'ci', 'item'), norm)
                    h_item_rate, _ = self.SEP_U(partition['ir'], graph, h, ('item', 'ir', 'rate'), ('rate', 'ri', 'item'), norm)
                    h_item_cate = F.softmax(self.cate_fc(h_item_cate), dim=1)
                    h_item_rate = F.softmax(self.rate_fc(h_item_rate), dim=1)
                if 'xmrec' in self.dataset:
                    h_item_bT, h_bT_idx = self.HGCNlayer(graph, h, ('item', 'ib', 'bT_idx'), ('bT_idx', 'bi', 'item'),
                                                         norm)
                    h_item_cpr, h_cpr_idx = self.HGCNlayer(graph, h, ('item', 'ip', 'cpr_idx'),
                                                           ('cpr_idx', 'pi', 'item'),
                                                           norm)
                if 'steam' in self.dataset:
                    h_user_age, h_age_idx = self.HGCNlayer(graph, h, ('user', 'ua', 'age'), ('age', 'au', 'user'), norm)
                    #h_user_age, h_age_idx = self.SEP_U(partition['u'], graph, h, ('user', 'ua', 'age'), ('age', 'au', 'user'), norm)
                    h_user_job, h_job_idx = self.HGCNlayer(graph, h, ('user', 'uj', 'job'), ('job', 'ju', 'user'), norm)
                    #h_user_job, h_job_idx = self.SEP_U(partition['u'], graph, h, ('user', 'uj', 'job'), ('job', 'ju', 'user'), norm)
            else:
                h_item_cate, h_item_rate, h_item_bT, h_item_cpr, h_user_age, h_user_job, h_bT_idx, h_cpr_idx, h_cate_idx, h_rate_idx, h_age_idx, h_job_idx = [], [], [], [], [], [], [], [], [], [], [], []
            if self.classify_as_edge:
                if 'xmrec' in self.dataset:
                    h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate, 'item_rate': h_item_rate,
                         'item_bT': h_item_bT, 'item_cpr': h_item_cpr, 'cate': h_cate_idx, 'rate': h_rate_idx,
                         'bT_idx': h_bT_idx, 'cpr_idx': h_cpr_idx}
                else:
                    h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate, 'item_rate': h_item_rate,
                         'user_age': h_user_age, 'user_job': h_user_job, 'cate': h_cate_idx, 'rate': h_rate_idx,
                         'age': h_age_idx, 'job': h_job_idx}
            else:
                if 'xmrec' in self.dataset:
                    h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate, 'item_rate': h_item_rate,
                         'item_bT': h_item_bT, 'item_cpr': h_item_cpr, 'bT_idx': h_bT_idx, 'cpr_idx': h_cpr_idx}
                else:
                    h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate, 'item_rate': h_item_rate,
                         'user_age': h_user_age, 'user_job': h_user_job, 'age': h_age_idx, 'job': h_job_idx}
        '''Transitional HyperConv'''
        #预训练融合部分不加图池化
        if pre_train and self.att_conv!=-1:
            '''TODO: xmrec_cn and steam - finetune transition strength'''
            h['item_edge'], h['user_edge'] = self.item_hyperedge, self.user_hyperedge
            if 'xmrec' in self.dataset:
                if self.att_conv == 0:
                    h_user_v2, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), -1,
                                                  self.trans_alpha,
                                                  h_item_cate + h_item_rate + h_item_bT + h_item_cpr, detach=True)
                elif self.att_conv == 2:
                    h_user_v2, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), -1,
                                                  self.trans_alpha, torch.cat((h_item_cate, h_item_rate,h_item_bT,h_item_cpr), dim=1),
                                                  detach=True,att=self.att_conv)
                else:
                    h_user_v2, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), -1,
                                                  self.trans_alpha,
                                                  torch.cat(
                                                      (torch.unsqueeze(h_item_cate, 1), torch.unsqueeze(h_item_rate, 1),
                                                       torch.unsqueeze(h_item_bT, 1), torch.unsqueeze(h_item_cpr, 1)
                                                       ),
                                                      dim=1), detach=True, att=self.att_conv)  #经过torch.unsqueeze后n*1*d矩阵，再cat后，是n*4*d矩阵
            elif 'steam' in self.dataset:
                if self.att_conv == 0:
                    h_user_v2, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), -1,
                                                  self.trans_alpha, h_item_cate + h_item_rate, detach=True)
                    h_item_v2, _ = self.HGCNlayer(graph, h, ('item', 'iu', 'user'), ('user', 'ui', 'item'), -1,
                                                  self.trans_alpha, h_user_age + h_user_job, detach=True)
                elif self.att_conv == 2:
                    h_user_v2, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), -1,
                                                  self.trans_alpha, torch.cat((h_item_cate,h_item_rate),dim=1), detach=True,att=self.att_conv)
                    h_item_v2, _ = self.HGCNlayer(graph, h, ('item', 'iu', 'user'), ('user', 'ui', 'item'), -1,
                                                  self.trans_alpha, torch.cat((h_user_age,h_user_job),dim=1), detach=True,att=self.att_conv)

                else:
                    h_user_v2, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), -1,
                                                  self.trans_alpha,
                                                  torch.cat(
                                                      (
                                                      torch.unsqueeze(h_item_cate, 1), torch.unsqueeze(h_item_rate, 1)),
                                                      dim=1), detach=True, att=self.att_conv)
                    h_item_v2, _ = self.HGCNlayer(graph, h, ('item', 'iu', 'user'), ('user', 'ui', 'item'), -1,
                                                  self.trans_alpha,
                                                  torch.cat(
                                                      (torch.unsqueeze(h_user_age, 1), torch.unsqueeze(h_user_job, 1)),
                                                      dim=1), detach=True, att=self.att_conv)
            if 'xmrec' in self.dataset:
                h_user = h_user + self.resid_beta * h_user_v2
                h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate, 'item_rate': h_item_rate,
                     'item_bT': h_item_bT, 'item_cpr': h_item_cpr, 'cate': h_cate_idx, 'rate': h_rate_idx,
                     'bT_idx': h_bT_idx, 'cpr_idx': h_cpr_idx}
            else:
                h_user = h_user + self.resid_beta * h_user_v2
                h_item = h_item + self.resid_beta * h_item_v2
                h = {'user': h_user, 'item': h_item, 'item_cate': h_item_cate,
                     'item_rate': h_item_rate,
                     'user_age': h_user_age, 'user_job': h_user_job, 'cate': h_cate_idx, 'rate': h_rate_idx,
                     'age': h_age_idx, 'job': h_job_idx}
        return h

    def create_ssl_loss_user(self, ssl_temp):
        # ssl_temp = 0.1
        h_user_v1 = torch.nn.functional.normalize(self.h_user_v1, p=2, dim=1)  #使用L2范数归一化
        h_user_v2 = torch.nn.functional.normalize(self.h_user_v2, p=2, dim=1)
        pos_score = torch.sum(torch.mul(h_user_v1, h_user_v2), dim=1)  #对应元素相乘后按行求和，pos_score[i]表示user_i消息v1和消息v2的内积
        neg_score = torch.matmul(h_user_v1, h_user_v2.T)  #neg_score[i,j]表示user_i的消息v1和user_j的消息v2的的内积，n*n
        pos_score = torch.exp(pos_score / ssl_temp)
        neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)  #neg_score[i]表示user_i的消息v1与所有user的消息v2内积之和
        ssl_loss = -torch.sum(torch.log(pos_score / neg_score))
        return ssl_loss

    def create_ssl_loss_item(self, ssl_temp):
        ssl_temp = 0.1
        h_item_v1 = torch.nn.functional.normalize(self.h_item_v1, p=2, dim=1)
        h_item_v2 = torch.nn.functional.normalize(self.h_item_v2, p=2, dim=1)
        # pos_score = torch.sum(torch.mul(h_item_v1, h_item_v2), dim=1)
        neg_score = torch.matmul(h_item_v1, h_item_v2.T)
        pos_score = 1 / ssl_temp
        neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)
        ssl_loss = -torch.sum(torch.log(pos_score / neg_score))
        return ssl_loss

    def create_ssl_loss_batched_item(self, ssl_temp, k=4, idx=0):
        ssl_temp = 0.1
        h_item_v1 = self.h_item_v1.split(self.h_item_v1.shape[0] // k + 1)[idx]  #分为k+1个batch，这为第idx个batch
        h_item_v2 = self.h_item_v2.split(self.h_item_v2.shape[0] // k + 1)[idx]
        h_item_v1 = torch.nn.functional.normalize(h_item_v1, p=2, dim=1)
        h_item_v2 = torch.nn.functional.normalize(h_item_v2, p=2, dim=1)
        neg_score = torch.matmul(h_item_v1, h_item_v2.T)
        pos_score = 1 / ssl_temp
        neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)
        ssl_loss = -torch.sum(torch.log(pos_score / neg_score))
        return ssl_loss

    def reweight_formula(self, beta, x, type=0):
        if type == 0:
            fx = (1 - beta) / (1 - beta ** x)
        elif type == 1:
            fx = 1 / (beta * x - beta + 1)
        elif type == 2:
            fx = 1 / torch.exp(beta * x - beta)
        elif type == 3:
            fx = 1 - torch.tanh(beta * x - beta)
        return torch.unsqueeze(fx, 1)

    def uniformity_loss(self, h_user):  #均匀性损失
        x = h_user[torch.randint(len(h_user), (2000,))]  #随机生成2000个索引，选择h_user中对应元素
        x = F.normalize(x, dim=-1)  #沿着最后一个维度归一化
        result = torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()  #torch.pdist(x,p=2)计算张量x中两两样本的欧几里得距离
        # print(result)
        # sys.exit()
        return 0.005 * result

    def create_bpr_loss(self, pos_g, neg_g, h, users_non_induct=None, loss_type='bpr'):
        if loss_type == 'align':
            # h['user'] = F.normalize(h['user'], dim=-1)
            # h['item'] = F.normalize(h['item'], dim=-1)
            sub_fig_feature = {'user': h['user'], 'item': h['item']}
            pos_score = self.pred.alignment_forward(pos_g, sub_fig_feature)  #边两端节点差值表示边对齐score
            mf_loss = pos_score[('user', 'ui', 'item')].norm(p=2, dim=1).pow(2).mean()  #按行归一化，平方，求平均
            # mf_loss = nn.Softplus()(pos_score[('user', 'ui', 'item')]).mean()
        elif loss_type == 'infonce':
            ssl_temp = 1e9
            # sub_fig_feature = {'user': F.normalize(h['user'],p=2, dim=1), 'item': F.normalize(h['item'],p=2, dim=1)}
            sub_fig_feature = {'user': h['user'], 'item': h['item']}
            pos_score = self.pred(pos_g, sub_fig_feature)
            pos_score = torch.exp(pos_score[('user', 'ui', 'item')] / ssl_temp)
            neg_score = self.pred(neg_g, sub_fig_feature)
            neg_score = neg_score[('user', 'ui', 'item')]
            neg_score = neg_score.reshape((pos_score.shape[0], self.neg_samples))
            neg_score = torch.exp(neg_score.sum(dim=1).unsqueeze(1) / ssl_temp)
            ssl_loss = -torch.sum(torch.log(pos_score / neg_score))
            mf_loss = ssl_loss
        elif loss_type == 'infoa':
            ssl_temp = 1e4
            # sub_fig_feature = {'user': F.normalize(h['user'],p=2, dim=1), 'item': F.normalize(h['item'],p=2, dim=1)}
            sub_fig_feature = {'user': h['user'], 'item': h['item']}
            pos_score = self.pred.alignment_forward(pos_g, sub_fig_feature)
            pos_score = torch.exp(pos_score[('user', 'ui', 'item')].norm(p=2, dim=1).pow(2) / ssl_temp)
            neg_score = self.pred.alignment_forward(neg_g, sub_fig_feature)
            neg_score = neg_score[('user', 'ui', 'item')].norm(p=2, dim=1).pow(2)
            neg_score = neg_score.reshape((pos_score.shape[0], self.neg_samples))
            neg_score = torch.exp(neg_score.sum(dim=1) / ssl_temp)
            ssl_loss = -torch.mean(torch.log(neg_score / pos_score))
            # ssl_loss = nn.Softplus()(pos_score - neg_score)
            mf_loss = ssl_loss.mean()
        elif loss_type == 'bpra':
            ssl_temp = 1e4
            # sub_fig_feature = {'user': F.normalize(h['user'],p=2, dim=1), 'item': F.normalize(h['item'],p=2, dim=1)}
            sub_fig_feature = {'user': h['user'], 'item': h['item']}
            pos_score = self.pred.alignment_forward(pos_g, sub_fig_feature)
            pos_score = pos_score[('user', 'ui', 'item')].norm(p=2, dim=1).pow(2).mean()
            neg_score = self.pred.alignment_forward(neg_g, sub_fig_feature)
            neg_score = neg_score[('user', 'ui', 'item')].norm(p=2, dim=1).pow(2).mean()
            ssl_loss = nn.Softplus()(pos_score - neg_score)
            mf_loss = ssl_loss
        elif loss_type == 'bpr_pos':
            sub_fig_feature = {'user': h['user'], 'item': h['item']}
            pos_score = self.pred(pos_g, sub_fig_feature)
            pos_score = pos_score[('user', 'ui', 'item')]
            mf_loss = nn.Softplus()(-pos_score)
            mf_loss = mf_loss.mean()
        else:
            sub_fig_feature = {'user': h['user'], 'item': h['item']}
            pos_score = self.pred(pos_g, sub_fig_feature)
            neg_score = self.pred(neg_g, sub_fig_feature)
            pos_score = pos_score[('user', 'ui', 'item')].repeat_interleave(self.neg_samples, dim=0)
            neg_score = neg_score[('user', 'ui', 'item')]
            mf_loss = nn.Softplus()(neg_score - pos_score)
            mf_loss = mf_loss.mean()
        if users_non_induct != None:
            regularizer = (1 / 2) * (self.user_embedding[users_non_induct].norm(2).pow(2) +
                                     self.item_embedding.norm(2).pow(2))
        else:
            regularizer = (1 / 2) * (self.user_embedding.norm(2).pow(2) +
                                     self.item_embedding.norm(2).pow(2))  #正则化项
        emb_loss = self.decay * regularizer

        # bpr_loss = self.ui_loss_alpha * mf_loss + emb_loss
        bpr_loss = mf_loss + emb_loss
        return bpr_loss, mf_loss, emb_loss

    '''TODO: bpr_loss for boughtTogether and ComparedTogether items link-prediction task'''

    def alignment(self, x, y):  #求x,y的对齐损失
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):  #求x的均匀性损失
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def create_ssl_loss_batched_group(self, ssl_temp, k=4, idx=0):
        ssl_temp = 0.1
        h_group_v1 = self.h_group_v1.split(self.h_group_v1.shape[0] // k + 1)[idx]
        h_group_v2 = self.h_group_v2.split(self.h_group_v2.shape[0] // k + 1)[idx]
        h_group_v1 = torch.nn.functional.normalize(h_group_v1, p=2, dim=1)
        h_group_v2 = torch.nn.functional.normalize(h_group_v2, p=2, dim=1)
        # pos_score = torch.sum(torch.mul(h_group_v1, h_group_v2), dim=1)
        neg_score = torch.matmul(h_group_v1, h_group_v2.T)
        pos_score = 1 / ssl_temp
        neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)
        ssl_loss = -torch.sum(torch.log(pos_score / neg_score))
        return ssl_loss

    def uniformity_loss_batched(self, x, k=4, idx=0):
        x = x.split(x.shape[0] // k + 1)[idx]
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_AU_loss(self, pos_g, h, k=4, idx=0):
        user_e, item_e = h['user'], h['item']
        user_e_batched = user_e.split(user_e.shape[0] // k + 1)[idx]
        item_e_batched = item_e.split(item_e.shape[0] // k + 1)[idx]
        related_item_e = item_e[pos_g.edges(etype='ui')[1]]
        related_item_e_batched = related_item_e.split(related_item_e.shape[0] // k + 1)[idx]
        align = self.alignment(user_e_batched, related_item_e_batched)
        uniform = (self.uniformity(user_e_batched) + self.uniformity(item_e_batched)) / 2
        loss = align + 1 * uniform
        return loss

    def create_item_bpr_loss(self, pos_g, neg_g, h, node_type='bT_idx', n_type=12, loss_type='bpr'):
        edge_dict = {'bT_idx': 'bi', 'cpr_idx': 'pi', 'cate': 'ci', 'rate': 'ri'}
        item_dict = {'bT_idx': 'item_bT', 'cpr_idx': 'item_cpr', 'cate': 'item_cate', 'rate': 'item_rate'}
        repeat_dict = {'bT_idx': self.neg_samples, 'cpr_idx': self.neg_samples, 'cate': n_type - 1, 'rate': n_type - 1}
        if loss_type == 'align':
            sub_fig_feature = {node_type: h[node_type], 'item': h[item_dict[node_type]]}
            pos_score = self.pred.alignment_forward(pos_g, sub_fig_feature)
            mf_loss = pos_score[(node_type, edge_dict[node_type], 'item')].norm(p=2, dim=1).pow(2).mean()
        else:
            sub_fig_feature = {node_type: h[node_type], 'item': h[item_dict[node_type]]}
            pos_score = self.pred(pos_g, sub_fig_feature)  #用边两端向量内积表示score
            neg_score = self.pred(neg_g, sub_fig_feature)
            # pos_score = pos_score[(node_type, edge_dict[node_type], 'item')].repeat_interleave(self.neg_samples, dim=0)
            pos_score = pos_score[(node_type, edge_dict[node_type], 'item')].repeat_interleave(repeat_dict[node_type],
                                                                                               dim=0)  #将pos_score重复，调成与neg大小一样
            neg_score = neg_score[(node_type, edge_dict[node_type], 'item')]
            mf_loss = nn.Softplus()(neg_score - pos_score) #log(1+exp(x))
            mf_loss = mf_loss.mean()
        regularizer = (1 / 2) * (h[node_type].norm(2).pow(2) +
                                 h[item_dict[node_type]].norm(2).pow(2))
        emb_loss = self.decay * regularizer

        bpr_loss = mf_loss + emb_loss
        return bpr_loss, mf_loss, emb_loss

    def create_user_bpr_loss(self, pos_g, neg_g, h, node_type='age', loss_type='bpr'):
        edge_dict = {'age': 'au', 'job': 'ju'}
        item_dict = {'age': 'user_age', 'job': 'user_job'}
        repeat_dict = {'age': self.neg_samples, 'job': self.neg_samples}
        if loss_type == 'align':
            sub_fig_feature = {node_type: h[node_type], 'user': h[item_dict[node_type]]}
            pos_score = self.pred.alignment_forward(pos_g, sub_fig_feature)
            mf_loss = pos_score[(node_type, edge_dict[node_type], 'user')].norm(p=2, dim=1).pow(2).mean()
        else:
            sub_fig_feature = {node_type: h[node_type], 'user': h[item_dict[node_type]]}
            # print(pos_g)
            pos_score = self.pred(pos_g, sub_fig_feature)
            neg_score = self.pred(neg_g, sub_fig_feature)
            # pos_score = pos_score[(node_type, edge_dict[node_type], 'item')].repeat_interleave(self.neg_samples, dim=0)
            pos_score = pos_score[(node_type, edge_dict[node_type], 'user')].repeat_interleave(repeat_dict[node_type],
                                                                                               dim=0)
            neg_score = neg_score[(node_type, edge_dict[node_type], 'user')]
            mf_loss = nn.Softplus()(neg_score - pos_score)
            mf_loss = mf_loss.mean()
        regularizer = (1 / 2) * (h[node_type].norm(2).pow(2) +
                                 h[item_dict[node_type]].norm(2).pow(2))
        emb_loss = self.decay * regularizer

        bpr_loss = mf_loss + emb_loss
        return bpr_loss, mf_loss, emb_loss

    def create_user_bpr_loss_v2(self, pos_g, neg_g, h, node_type='age', n_type=7):
        edge_dict = {'age': 'au', 'job': 'ju'}
        user_dict = {'age': 'user_age', 'job': 'user_job'}
        repeat_dict = {'age': n_type - 1, 'job': n_type - 1}
        sub_fig_feature = {node_type: h[node_type], 'user': h[user_dict[node_type]]}
        pos_score = self.pred(pos_g, sub_fig_feature)
        neg_score = self.pred(neg_g, sub_fig_feature)
        pos_score = pos_score[(node_type, edge_dict[node_type], 'user')].repeat_interleave(repeat_dict[node_type],
                                                                                           dim=0)
        neg_score = neg_score[(node_type, edge_dict[node_type], 'user')]
        mf_loss = nn.Softplus()(neg_score - pos_score)
        mf_loss = mf_loss.mean()
        regularizer = (1 / 2) * (h[node_type].norm(2).pow(2) +
                                 h[user_dict[node_type]].norm(2).pow(2))
        emb_loss = self.decay * regularizer

        bpr_loss = mf_loss + emb_loss
        return bpr_loss, mf_loss, emb_loss

    def create_classify_loss(self, item_embedding, label): #j交叉熵损失，衡量模型输出与真实标签之间差异
        return F.cross_entropy(item_embedding, label, reduction='mean')
