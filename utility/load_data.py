import sys
sys.path.append('/workspace/UPRTH')
import numpy as np
import random as rd
import dgl
from silearn.graph import GraphSparse
from silearn.optimizer.enc.partitioning.propagation import OperatorPropagation
from silearn.model.encoding_tree import Partitioning
from torch_scatter import scatter_sum
import torch
import pickle
import os
import scipy.sparse as sp


class Data(object):
    def __init__(self, path, batch_size, dataset, se, device):
        self.path = path
        self.batch_size = batch_size
        self.device = device
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        ui_file = path + '/user_item.txt'
        ic_file = path + '/item_category.txt'
        ir_file = path + '/item_rating.txt'
        bT_file = path + '/boughtTogether.txt'
        cpr_file = path + '/comparedTogether.txt'
        ua_file = path + '/user_friend.txt'
        uo_file = path + '/user_groups.txt'

        # get number of users and items
        self.n_users, self.n_items, self.n_cate, self.n_rate = 0, 0, 0, 0
        self.n_age,self.n_job = 0,0
        self.n_train, self.n_test, self.ic_interactions, self.ir_interactions = 0, 0, 0, 0
        self.exist_users = []

        user_item_src, user_item_dst = [],[]  #存放train的uid和item id（即user-item对）
        item_cate_src, item_cate_dst = [],[]  #存放item和cate（即item-cate/类别对）
        item_rate_src,item_rate_dst = [],[]  #存放item和rate（即item-rate/等级对）
        item_bT_src,item_bT_dst = [],[]
        item_cpr_src,item_cpr_dst = [],[]
        user_age_src,user_age_dst = [],[]  #存放uid和friend_idx（即uid-friend_idx对）
        user_job_src,user_job_dst = [],[]  #存放uid和group（即uid-group群组对）

        self.train_items, self.test_set = {}, {}
        with open(train_file, 'r') as f:
            line = f.readline().strip()
            while line != '':
                line = line.strip('\n').split(' ')
                uid = int(line[0])
                self.exist_users.append(uid)
                items = [int(g) for g in line[1:]]
                self.train_items[uid] = items
                self.n_users = max(self.n_users, uid)
                self.n_items = max(self.n_items, max(items))
                self.n_train += len(items)  #train的user-item对数目
                for g in line[1:]:
                    user_item_src.append(uid)
                    user_item_dst.append(int(g))
                line = f.readline().strip()

        with open(test_file, 'r') as f:
            line = f.readline().strip()
            while line != '':
                line = line.strip('\n').split(' ')
                uid = int(line[0])
                items_test = [int(g) for g in line[1:]]
                self.test_set[uid] = items_test
                self.n_items = max(self.n_items, max(items_test))  #更新item的数目，test里的item也算进来
                self.n_test += len(items_test)  #test的user-item对数目
                line = f.readline().strip()
        self.n_users += 1
        self.n_items += 1  #因为id都是从0开始的，算总数时编号+1

        with open(ic_file, 'r') as f:
            line = f.readline().strip()
            while line != '':
                line = line.strip('\n').split(' ')
                self.ic_interactions += 1
                iid = int(line[0])
                if 'xmrec' in dataset:
                    category = int(line[1])
                    self.n_cate = max(self.n_cate, category)
                    item_cate_src.append(iid)
                    item_cate_dst.append(category)
                else:
                    for cate in line[1:]:
                        cate = int(cate)
                        self.n_cate = max(self.n_cate, cate)
                        item_cate_src.append(iid)
                        item_cate_dst.append(cate)
                line = f.readline().strip()
        self.n_cate += 1 #item的类别cate数目

        with open(ir_file, 'r') as f:
            line = f.readline().strip()
            while line != '':
                line = line.strip('\n').split(' ')
                self.ir_interactions += 1
                iid = int(line[0])
                if 'xmrec' in dataset:
                    rate = int(eval(line[1])*2-2)
                    item_rate_src.append(iid)
                    item_rate_dst.append(rate)
                else:
                    for rate in line[1:]:
                        rate = int(rate)
                        item_rate_src.append(iid)
                        item_rate_dst.append(rate)
                        self.n_rate = max(self.n_rate, rate)
                line = f.readline().strip()
            if 'xmrec' in dataset:
                self.n_rate = 9
            else:
                self.n_rate += 1  #item的等级rate数目

        if 'xmrec' in dataset:
            bT_idx = 0
            with open(bT_file, 'r') as f:
                line = f.readline().strip()
                while line != '':
                    line = line.strip('\n').split(' ')
                    for iid in line:
                        item_bT_src.append(int(iid))
                        item_bT_dst.append(bT_idx)
                    bT_idx += 1
                    line = f.readline().strip()
            cpr_idx = 0
            with open(cpr_file, 'r') as f:
                line = f.readline().strip()
                while line != '':
                    line = line.strip('\n').split(' ')
                    for iid in line:
                        item_cpr_src.append(int(iid))
                        item_cpr_dst.append(cpr_idx)
                    cpr_idx += 1
                    line = f.readline().strip()

            self.n_cluster_bT = bT_idx
            self.n_cluster_cpr = cpr_idx
        elif 'steam' == dataset:
            friend_idx = 0
            with open(ua_file, 'r') as f:
                line = f.readline().strip()
                while line != '':
                    line = line.strip('\n').split(' ')
                    for uid in line:
                        user_age_src.append(int(uid))
                        user_age_dst.append(friend_idx)  #这一line中的uid有共同朋友friend_idx/处于相同年龄段
                    friend_idx += 1
                    line = f.readline().strip()
            self.n_age = friend_idx  #朋友类别数

            with open(uo_file, 'r') as f:
                line = f.readline().strip()
                while line != '':
                    line = line.strip('\n').split(' ')
                    # self.ua_interactions += 1
                    uid = int(line[0])
                    groups = line[1:]
                    for g in groups:
                        g = int(g)
                        self.n_job = max(self.n_job, g)  #更新最大group编号
                        user_job_src.append(uid)
                        user_job_dst.append(g)
                    line = f.readline().strip()
            self.n_job += 1  #uid的group总数

        self.print_statistics(dataset)

        self.item_cate_idx = item_cate_src
        self.item_rate_idx = item_rate_src
        self.cate_label = item_cate_dst
        self.rate_label = item_rate_dst

        if 'xmrec' in dataset:
            data_dict = {
                ('user', 'ui', 'item'): (user_item_src, user_item_dst),
                ('item', 'iu', 'user'): (user_item_dst, user_item_src),
                ('item', 'ic', 'cate'): (item_cate_src, item_cate_dst),
                ('cate', 'ci', 'item'): (item_cate_dst, item_cate_src),
                ('item', 'ir', 'rate'): (item_rate_src, item_rate_dst),
                ('rate', 'ri', 'item'): (item_rate_dst, item_rate_src),
                ('item', 'ib', 'bT_idx'): (item_bT_src, item_bT_dst),
                ('bT_idx', 'bi', 'item'): (item_bT_dst, item_bT_src),
                ('item', 'ip', 'cpr_idx'): (item_cpr_src, item_cpr_dst),
                ('cpr_idx', 'pi', 'item'): (item_cpr_dst, item_cpr_src),
            }
            num_dict = {
                'user': self.n_users, 'item': self.n_items, 'cate': self.n_cate, 'rate': self.n_rate,
                'bT_idx': self.n_cluster_bT,
                'cpr_idx': self.n_cluster_cpr,
            }
            self.g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)
            if se==1:
                '''
                if os.path.exists(path+'/partition_3D.pickle'):
                    self.partition_ui,self.partition_iu,self.partition_ic,self.partition_ir,self.partition_ib,self.partition_ip=pickle.load(open(path+'/partition_3D.pickle',"rb+"))
                else:
                    #partition为一个元组（[S1,src,dst],[S2,src,dst])，S1是个list，S1[i]为用户i对应社区，src,dst表示聚合后的二部图源社区，目标节点对
                    self.partition_ui=self.partition(self.n_users,self.n_items,user_item_src,user_item_dst)
                    self.partition_iu=self.partition(self.n_items,self.n_users,user_item_dst,user_item_src)
                    self.partition_ic=self.partition(self.n_items,self.n_cate,item_cate_src, item_cate_dst)
                    self.partition_ir=self.partition(self.n_items,self.n_rate,item_rate_src, item_rate_dst)
                    self.partition_ib=self.partition(self.n_items,self.n_cluster_bT,item_bT_src, item_bT_dst)
                    self.partition_ip=self.partition(self.n_items,self.n_cluster_cpr,item_cpr_src, item_cpr_dst)
                    with open(path+'/partition_3D.pickle',"wb")as f:
                        pickle.dump((self.partition_ui,self.partition_iu,self.partition_ic,self.partition_ir,self.partition_ib,self.partition_ip),f)
                    print('partition down.')
                '''
                if os.path.exists(path+'/partition_5D.pickle'):
                    self.partition_ui,self.partition_iu,self.partition_ic,self.partition_ir,self.partition_ib,self.partition_ip=pickle.load(open(path+'/partition_5D.pickle',"rb+"))
                else:
                    #partition为一个元组（[S1,src,dst],[S2,src,dst])，S1是个list，S1[i]为用户i对应社区，src,dst表示聚合后的二部图源社区，目标节点对
                    self.partition_ui=self.partition_high(self.n_users,self.n_items,user_item_src,user_item_dst)
                    self.partition_iu=self.partition_high(self.n_items,self.n_users,user_item_dst,user_item_src)
                    self.partition_ic=self.partition_high(self.n_items,self.n_cate,item_cate_src, item_cate_dst)
                    self.partition_ir=self.partition_high(self.n_items,self.n_rate,item_rate_src, item_rate_dst)
                    self.partition_ib=self.partition_high(self.n_items,self.n_cluster_bT,item_bT_src, item_bT_dst)
                    self.partition_ip=self.partition_high(self.n_items,self.n_cluster_cpr,item_cpr_src, item_cpr_dst)
                    with open(path+'/partition_5D.pickle',"wb")as f:
                        pickle.dump((self.partition_ui,self.partition_iu,self.partition_ic,self.partition_ir,self.partition_ib,self.partition_ip),f)
                    print('partition down.')
                '''
                if os.path.exists(path+'/partition_multi_ui.pickle'):
                    self.partition_i, self.partition_u=pickle.load(open(path+'/partition_multi_ui.pickle',"rb+"))
                else:
                    self.partition_i=self.partition(self.n_items, self.n_users+self.n_cate+self.n_rate+self.n_cluster_bT+self.n_cluster_cpr, user_item_dst+item_cate_src+item_rate_src+item_bT_src+item_cpr_src, user_item_src+[i+self.n_users for i in item_cate_dst]+[i+self.n_users+self.n_cate for i in item_rate_dst]+[i+self.n_users+self.n_cate+self.n_rate for i in item_bT_dst]+[i+self.n_users+self.n_cate+self.n_rate+self.n_cluster_bT for i in item_cpr_dst])
                    with open(path+'/partition_multi_ui.pickle',"wb")as f:
                        pickle.dump((self.partition_i,self.partition_ui),f)
                    print('partition multi-i and multi-u down.')
                '''

        elif 'steam' == dataset:
            data_dict = {
                ('user', 'ui', 'item'): (user_item_src, user_item_dst),
                ('item', 'iu', 'user'): (user_item_dst, user_item_src),
                ('item', 'ic', 'cate'): (item_cate_src, item_cate_dst),
                ('cate', 'ci', 'item'): (item_cate_dst, item_cate_src),
                ('item', 'ir', 'rate'): (item_rate_src, item_rate_dst),
                ('rate', 'ri', 'item'): (item_rate_dst, item_rate_src),
                ('user', 'ua', 'age'): (user_age_src, user_age_dst),
                ('age', 'au', 'user'): (user_age_dst, user_age_src),
                ('user', 'uj', 'job'): (user_job_src, user_job_dst),
                ('job', 'ju', 'user'): (user_job_dst, user_job_src),
            }
            num_dict = {
                'user': self.n_users, 'item': self.n_items, 'cate': self.n_cate, 'rate': self.n_rate,
                'age': self.n_age,
                'job': self.n_job,
            }
            self.g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)
            self.age_label = user_age_dst
            self.job_label = user_job_dst

            if se==1:
                if os.path.exists(path+'/partition_5D.pickle'):
                    self.partition_ui,self.partition_iu,self.partition_ic,self.partition_ir,self.partition_ua,self.partition_uj=pickle.load(open(path+'/partition_5D.pickle',"rb+"))
                else:
                    #partition为一个元组（[S1,src,dst],[S2,src,dst],...)，S1是个list，S1[i]为用户i对应社区，src,dst表示聚合后的二部图源社区，目标节点对
                    self.partition_ui=self.partition_high(self.n_users,self.n_items,user_item_src,user_item_dst)
                    self.partition_iu=self.partition_high(self.n_items,self.n_users,user_item_dst,user_item_src)
                    self.partition_ic=self.partition_high(self.n_items,self.n_cate,item_cate_src, item_cate_dst)
                    self.partition_ir=self.partition_high(self.n_items,self.n_rate,item_rate_src, item_rate_dst)
                    self.partition_ua=self.partition_high(self.n_users,self.n_age,user_age_src, user_age_dst)
                    self.partition_uj=self.partition_high(self.n_users,self.n_job,user_job_src, user_job_dst)
                    with open(path+'/partition_5D.pickle',"wb")as f:
                        pickle.dump((self.partition_ui,self.partition_iu,self.partition_ic,self.partition_ir,self.partition_ua,self.partition_uj),f)
                    print('partition down.')
                '''
                if os.path.exists(path+'/partition_multi_ui.pickle'):
                    self.partition_i, self.partition_u=pickle.load(open(path+'/partition_multi_ui.pickle',"rb+"))
                else:
                    self.partition_i=self.partition(self.n_items, self.n_users+self.n_cate+self.n_rate, user_item_dst+item_cate_src+item_rate_src, user_item_src+[i+self.n_users for i in item_cate_dst]+[i+self.n_users+self.n_cate for i in item_rate_dst])
                    self.partition_u=self.partition(self.n_users, self.n_items+self.n_age+self.n_job, user_item_src+user_age_src+user_job_src, user_item_dst+[i+self.n_items for i in user_age_dst]+[i+self.n_items+self.n_age for i in user_job_dst])
                    with open(path+'/partition_multi_ui.pickle',"wb")as f:
                        pickle.dump((self.partition_i,self.partition_u),f)
                    print('partition multi-i and multi-u down.')
                '''
        
        
    def partition(self, src_num, dst_num, src, dst):

        def encoding2tree(adj_matrix):
            #encoding tree
            num=adj_matrix.shape[0]
            edges = np.array(adj_matrix.nonzero()) # [2, E]
            ew = adj_matrix[edges[0, :], edges[1, :]]
            devices = self.device
            ew, edges = torch.tensor(ew, device=devices), torch.tensor(edges, device=devices).t()
            dist = scatter_sum(ew, edges[:, 1]) + scatter_sum(ew, edges[:, 0])
            dist = dist / (2 * ew.sum())
            print('construct encoding tree with num:{}...'.format(num))
            g = GraphSparse(edges, ew, dist)
            optim = OperatorPropagation(Partitioning(g, None))
            optim.perform(p=0.2)
            division = optim.enc.node_id
            totol_comm = torch.max(division) + 1
            print('construct encoding tree done with comm:{}'.format(totol_comm))
            return division, totol_comm
    
        adj_matrix_A=np.zeros((src_num, dst_num))
        adj_matrix_A[src,dst]=1  #二部图邻接矩阵
        #adj_matrix=np.dot(adj_matrix_A,adj_matrix_A.T) 
        D_A_r=np.sum(adj_matrix_A,axis=0)
        nonzero_mask = D_A_r != 0
        D_A_r[nonzero_mask]=1.0/np.log2(D_A_r[nonzero_mask]+1) #np.sqrt() np.log2()
        D_A_r=np.diag(D_A_r)#二部图邻接矩阵中dst节点的度矩阵
        adj_matrix=np.dot(np.dot(adj_matrix_A,D_A_r),adj_matrix_A.T)
        adj_matrix -= np.diag(np.diag(adj_matrix))
        #D_v=np.sum(adj_matrix,axis=0) #adj_matrix中节点的度矩阵
        #nonzero_mask = D_v != 0
        #D_v[nonzero_mask]=1.0/np.sqrt(D_v[nonzero_mask])
        #D_v=np.diag(D_v)
        #adj_matrix=np.dot(np.dot(D_v,adj_matrix),D_v)
        index=np.where(np.sum(adj_matrix,axis=1)==0)
        for i in index:
            adj_matrix[i,i]=0.001  #处理孤立节点
        print(np.count_nonzero(adj_matrix))
        division1, totol_comm1= encoding2tree(adj_matrix)

        comm_matrix_S=np.zeros((src_num, totol_comm1)) #社区聚合矩阵,每个用户去了哪个社区
        comm_matrix_S[list(range(src_num)),division1.tolist()]=1
        adj_matrix_A=np.dot(comm_matrix_S.T,adj_matrix_A)  #新的二部图邻接矩阵
        edge_index_A=np.array(adj_matrix_A.nonzero())
        src_new1,dst_new1=edge_index_A[0].tolist(),edge_index_A[1].tolist()

        if totol_comm1==src_num:
            return ([division1.tolist(),src_new1,dst_new1],[division1.tolist(),src_new1,dst_new1])
        else:
            #adj_matrix = np.dot(adj_matrix_A,adj_matrix_A.T) 
            adj_matrix = np.dot(np.dot(adj_matrix_A,D_A_r),adj_matrix_A.T)
            adj_matrix -= np.diag(np.diag(adj_matrix))
            #D_v=np.sum(adj_matrix,axis=0) #adj_matrix中节点的度矩阵
            #nonzero_mask = D_v != 0
            #D_v[nonzero_mask]=1.0/np.sqrt(D_v[nonzero_mask])
            #D_v=np.diag(D_v)
            #adj_matrix=np.dot(np.dot(D_v,adj_matrix),D_v)
            index=np.where(np.sum(adj_matrix,axis=1)==0)
            for i in index:
                adj_matrix[i,i]=0.001  #处理孤立社区
            print(np.count_nonzero(adj_matrix))
            division2, totol_comm2= encoding2tree(adj_matrix)

            comm_matrix_S=np.zeros((totol_comm1, totol_comm2)) #社区聚合矩阵,每个用户去了哪个社区
            comm_matrix_S[list(range(totol_comm1)),division2.tolist()]=1
            adj_matrix_A=np.dot(comm_matrix_S.T,adj_matrix_A)  #新的二部图邻接矩阵
            edge_index_A=np.array(adj_matrix_A.nonzero())
            src_new2,dst_new2=edge_index_A[0].tolist(),edge_index_A[1].tolist()
            return ([division1.tolist(),src_new1,dst_new1],[division2.tolist(),src_new2,dst_new2])

    def partition_high(self, src_num, dst_num, src, dst, num_layer=5):

        def encoding2tree(adj_matrix):
            #encoding tree
            num=adj_matrix.shape[0]
            #edges = np.array(adj_matrix.nonzero()) # [2, E]
            #ew = adj_matrix[edges[0, :], edges[1, :]]
            edges = np.vstack(adj_matrix.nonzero())
            ew = np.array([adj_matrix[edges[0, i], edges[1, i]] for i in range(edges.shape[1])]).flatten()
            devices = self.device
            ew, edges = torch.tensor(ew, device=devices), torch.tensor(edges, device=devices, dtype=torch.int64).t()
            dist = scatter_sum(ew, edges[:, 1]) + scatter_sum(ew, edges[:, 0])
            dist = dist / (2 * ew.sum())
            print('construct encoding tree with num:{}...'.format(num))
            g = GraphSparse(edges, ew, dist)
            optim = OperatorPropagation(Partitioning(g, None))
            optim.perform(p=0.2)
            division = optim.enc.node_id
            totol_comm = torch.max(division) + 1
            print('construct encoding tree done with comm:{}'.format(totol_comm))
            return division, totol_comm
    
        #adj_matrix_A=np.zeros((src_num, dst_num))
        #adj_matrix_A[src,dst]=1  #二部图邻接矩阵
        adj_matrix_A = sp.lil_matrix((src_num, dst_num))
        adj_matrix_A[src, dst] = 1  # 二部图邻接矩阵
        #D_A_r=np.sum(adj_matrix_A,axis=0)
        D_A_r = np.array(adj_matrix_A.sum(axis=0)).flatten()
        nonzero_mask = D_A_r != 0
        D_A_r[nonzero_mask]=1.0/np.log2(D_A_r[nonzero_mask]+1) #np.sqrt() np.log2()
        D_A_r=np.diag(D_A_r)#二部图邻接矩阵中dst节点的度矩阵
        D_A_r = sp.lil_matrix(D_A_r)

        out=[]
        old_num=src_num
        for layer in range(num_layer):
            #adj_matrix=np.dot(np.dot(adj_matrix_A,D_A_r),adj_matrix_A.T)
            #adj_matrix -= np.diag(np.diag(adj_matrix))
            #index=np.where(np.sum(adj_matrix,axis=1)==0)
            adj_matrix = adj_matrix_A.tocsr().dot( D_A_r.tocsr()).dot(adj_matrix_A.transpose().tocsr())
            adj_matrix=adj_matrix.tolil()
            adj_matrix.setdiag(0)  # 去除对角元素
            index = np.where(adj_matrix.sum(axis=1) == 0)[0]
            for i in index:
                adj_matrix[i,i]=0.001  #处理孤立节点
            #print(np.count_nonzero(adj_matrix))
            print(adj_matrix.nnz)
            division, totol_comm= encoding2tree(adj_matrix)

            #comm_matrix_S=np.zeros((old_num, totol_comm)) #社区聚合矩阵,每个用户去了哪个社区
            #comm_matrix_S[list(range(old_num)),division.tolist()]=1
            comm_matrix_S = sp.lil_matrix((old_num, totol_comm))
            comm_matrix_S[list(range(old_num)), division.tolist()] = 1
            #adj_matrix_A=np.dot(comm_matrix_S.T,adj_matrix_A)  #新的二部图邻接矩阵
            adj_matrix_A = comm_matrix_S.T @ adj_matrix_A
            edge_index_A=np.array(adj_matrix_A.nonzero())
            src_new1,dst_new1=edge_index_A[0].tolist(),edge_index_A[1].tolist()
            if totol_comm==old_num:
                print('early stop at layer:',layer)
                for _ in range(num_layer-layer):
                    out.extend([[division.tolist(),src_new1,dst_new1]])
                return tuple(out)
            else:
                old_num=totol_comm
                out.extend([[division.tolist(),src_new1,dst_new1]])
        return tuple(out)
        
    def print_statistics(self, dataset):
        print('n_users=%d, n_items=%d, n_cate=%d, n_rate=%d' % (
        self.n_users, self.n_items, self.n_cate, self.n_rate))
        if 'xmrec' in dataset:
            print('n_cluster_bT=%d, n_cluster_cpr=%d'%(self.n_cluster_bT, self.n_cluster_cpr))
        elif 'steam' == dataset:
            print('n_ages=%d,n_jobs=%d'%(self.n_age,self.n_job))
        print('n_ui_interactions=%d, n_ic_interactions=%d,n_ir_interactions=%d' % (
            self.n_train + self.n_test, self.ic_interactions, self.ir_interactions))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)  #从训练的userid中随机采样batch_size个用户id
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]  #有放回的采样batch_size个用户id

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]  #u的历史交互item
            n_pos_items = len(pos_items)  #u的历史交互item数
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]  #随机抽样一个u的历史交互item

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)  #从所有item中随机抽样未交互过的item
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items  #batch_size下的用户id列表，对user随机抽样的正负交互item列表，每个user一个item（正：有历史交互、负：无历史交互）

    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)  #随机抽样batch_size个test中的uid
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]  #在现有/train中有放回的抽样batch_size个uid

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u] + self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id) ##从所有item中随机抽样train和test中均未交互过的item
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
