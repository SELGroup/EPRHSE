import random
import sys

import torch
import math
import torch.optim as optim
from utility.load_data import *
from utility.parser import *
from utility.batch_test import *
from utility.helper import early_stopping, random_batch_users, ensureDir, convert_dict_list, convert_list_str
from baseline_models import *
from time import time
import numpy as np
import dgl
import os
import json

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
def get_edges_induct(g, users_induct):
    # print(g.edges(etype = 'ui'))
    # print(g.edges(etype = 'iu'))
    idx = np.in1d(g.edges(etype='ui')[0], users_induct)
    # print(idx.nonzero()[0])
    # sys.exit()
    return idx.nonzero()[0]


def main(args):
    # Step 1: Prepare graph data and device ================================================================= #
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    users_to_test = list(data_generator.test_set.keys())
    g = data_generator.g
    r = 1/args.induct_ratio
    if args.inductive:
        original_g = g.to(device)
        users_induct = users_to_test[len(users_to_test) - math.ceil(len(users_to_test) // r):]
        users_non_induct = [u for u in list(range(data_generator.n_users)) if u not in users_induct]
        # print(len(users_non_induct))
        users_transduct = users_to_test[:len(users_to_test) - math.ceil(len(users_to_test) // r)]
        edges_induct = get_edges_induct(g, users_induct)
        # print(g.num_edges(etype = 'ui'),g.num_edges(etype = 'iu'))
        g_inductive = dgl.remove_edges(g, edges_induct, etype='ui')
        # print(g_inductive.num_edges(etype = 'ui'))
        g_inductive = dgl.remove_edges(g_inductive, edges_induct, etype='iu')
        # print(g_inductive.num_edges(etype = 'iu'))
        # print(g_inductive.nodes(ntype='user'))
        # print(g_inductive.ndata[dgl.NID]['user'])
        # inductive_map = dict(zip(g_inductive.ndata[dgl.NID]['user'].tolist(), g_inductive.nodes(ntype='user').tolist()))
        # users_transduct = [inductive_map[u] for u in users_to_test[:len(users_to_test)//2]]
        # print(inductive_map)
        # print(users_transduct)
        # sys.exit()
        g = g_inductive
    item_cate_idx = torch.LongTensor(data_generator.item_cate_idx).to(device)
    item_rate_idx = torch.LongTensor(data_generator.item_rate_idx).to(device)
    cate_label = torch.LongTensor(data_generator.cate_label).to(device)
    rate_label = torch.LongTensor(data_generator.rate_label).to(device)
    if 'ml-1m' in args.dataset:
        age_label = torch.LongTensor(data_generator.age_label).to(device)
        job_label = torch.LongTensor(data_generator.job_label).to(device)
    g = g.to(device)
    pos_g = construct_user_item_bigraph(g)

    if args.attrimask:
        args.classify_as_edge = 0
        args.item_pretrain = 0
        args.user_pretrain = 0
        args.att_conv = -1
    if args.item_pretrain and 'xmrec' in args.dataset:
        pos_g_bT = construct_item_related_bigraph(g, 'bT_idx')
        pos_g_cpr = construct_item_related_bigraph(g, 'cpr_idx')
    if args.user_pretrain and 'steam' in args.dataset:
        pos_g_age = construct_user_related_bigraph(g, 'age')
        pos_g_job = construct_user_related_bigraph(g, 'job')
    if args.classify_as_edge:
        pos_g_c2e_cate = construct_item_related_bigraph(g, 'cate')
        pos_g_c2e_rate = construct_item_related_bigraph(g, 'rate')
    if args.gat != 0:
        model = GAT(args, g, args.embed_size, 8, args.embed_size, device).to(device)
    else:
        model = LightGCN(args, g, device).to(device)

    t0 = time()
    cur_best_pre_0, stopping_step = 0, 0
    bpr_loss_bT, bpr_loss_cpr, bpr_loss_age, bpr_loss_job = 0, 0, 0, 0
    pre_train_best = float('inf')
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    rec_loger_induct, ndcg_loger_induct = [], []
    rec_loger_transduct, ndcg_loger_transduct = [], []
    recall_split_loger, ndcg_split_loger = [], []
    user_uniform_loss, item_uniform_loss = 0, 0
    ui_loss_alpha = eval(args.regs)[0]
    # From 0 to 1
    # optimizer = optim.Adam(model.parameters(), lr=args.pre_lr)
    if args.pre_train:
        optimizer = optim.Adam(model.parameters(), lr=args.pre_lr)
        if args.attrimask==1:
            args.epoch = 500
        for epoch in range(args.epoch):
            t1 = time()
            t_pre = time()
            neg_g = construct_negative_graph(g, args.neg_samples, device=device)
            # trans_conv = True
            trans_conv = False if (args.gcc or args.sgl) else True
            if args.sgl == 0:
                embedding_h = model(g, True, trans_conv)
            bpr_loss = 0
            if args.sgl:
                transform = dgl.DropEdge(p=0.1)
                transform_2 = dgl.DropEdge(p=0.1)
                embedding_h = model.lightgcn_forward(g)
                g_v1 = transform(g)
                g_v2 = transform_2(g)
                embedding_h_v1 = model.lightgcn_forward(g_v1)
                embedding_h_v2 = model.lightgcn_forward(g_v2)
                '''k=4 for cn, 64 for mx, 64 for steam'''
                cl_loss_user = model.create_ssl_loss_user(embedding_h_v1['user'], embedding_h_v2['user'], ssl_temp=0.5,
                                                          ssl_reg=1e-7,
                                                          k=64)
                cl_loss_item = model.create_ssl_loss_user(embedding_h_v1['item'], embedding_h_v2['item'], ssl_temp=0.5,
                                                          ssl_reg=1e-7,
                                                          k=16)
                bpr_loss, mf_loss, emb_loss = model.create_bpr_loss(pos_g, neg_g, embedding_h, loss_type=args.loss)
                ttl_loss = bpr_loss + cl_loss_user + cl_loss_item
            elif args.gcc:
                ctr_list = ['user', 'item', 'item_rate', 'item_cate', 'item_bT',
                            'item_cpr'] if 'xmrec' in args.dataset else ['user', 'item', 'item_rate', 'item_cate',
                                                                         'user_age', 'user_job']
                transform = dgl.DropEdge(p=0.2)
                g_v2 = transform(g)
                embedding_h_v2 = model(g_v2, True, trans_conv)
                cl_loss = 0
                for k in ctr_list:
                    # print(embedding_h[k].shape,embedding_h[k].shape[0] // 10)
                    cl_loss += model.create_ssl_loss_user(embedding_h[k], embedding_h_v2[k], ssl_temp=0.1, ssl_reg=1e-7,
                                                          k=64, reg=1e-7)
                ttl_loss = cl_loss
            if args.gcc or args.sgl:
                optimizer.zero_grad()
                ttl_loss.backward()
                optimizer.step()
                if (epoch + 1) % (args.verbose * 10) != 0:
                    if args.verbose > 0 and epoch % (args.verbose * 10) == 0:
                        perf_str = 'Epoch %d [%.1fs]: train==[%.5f]' % (
                            epoch, time() - t_pre, ttl_loss)
                        print(perf_str)
                    continue
                pre_train_best, stopping_step, should_stop = early_stopping(ttl_loss, pre_train_best,
                                                                            stopping_step, expected_order='dec',
                                                                            flag_step=args.flag_step)
                if should_stop == True:
                    # if epoch == 99:
                    print('Pre-train stopped.')
                    break
                continue
            if args.pre_train_task not in [0, 5]:
                if args.inductive:
                    bpr_loss, mf_loss, emb_loss = model.create_bpr_loss(pos_g, neg_g, embedding_h, users_non_induct,
                                                                        loss_type=args.loss)
                else:
                    bpr_loss, mf_loss, emb_loss = model.create_bpr_loss(pos_g, neg_g, embedding_h, loss_type=args.loss)
            if args.loss == 'auloss':
                user_uniform_loss = model.uniformity_loss(embedding_h['user'])
                item_uniform_loss = model.uniformity_loss(embedding_h['item'])
            if args.classify_as_edge:
                neg_g_c2e_cate = construct_negative_item_graph_c2ep(g, data_generator.n_cate, device, 'cate')
                neg_g_c2e_rate = construct_negative_item_graph_c2ep(g, data_generator.n_rate, device, 'rate')
                if args.pre_train_task == 1:
                    cate_loss, _, _ = model.create_item_bpr_loss(pos_g_c2e_cate, neg_g_c2e_cate, embedding_h, 'cate',
                                                                 data_generator.n_cate)
                    rate_loss, _, _ = model.create_item_bpr_loss(pos_g_c2e_rate, neg_g_c2e_rate, embedding_h, 'rate',
                                                                 data_generator.n_rate)
                elif args.pre_train_task == 2:
                    cate_loss = 0
                    rate_loss, _, _ = model.create_item_bpr_loss(pos_g_c2e_rate, neg_g_c2e_rate, embedding_h, 'rate',
                                                                 data_generator.n_rate)
                elif args.pre_train_task == 3:
                    cate_loss, _, _ = model.create_item_bpr_loss(pos_g_c2e_cate, neg_g_c2e_cate, embedding_h, 'cate',
                                                                 data_generator.n_cate)
                    rate_loss = 0
                elif args.pre_train_task == 4:
                    cate_loss = 0
                    rate_loss = 0
            else:
                if embedding_h['item_cate'].shape[0] != cate_label.shape:
                    embedding_h['item_cate'] = torch.index_select(embedding_h['item_cate'], 0, item_cate_idx)
                if embedding_h['item_rate'].shape[0] != rate_label.shape:
                    embedding_h['item_rate'] = torch.index_select(embedding_h['item_rate'], 0, item_rate_idx)
                if args.pre_train_task == 1:
                    # print(embedding_h['item_cate'].shape,cate_label.shape)
                    # item_cate = torch.index_select(embedding_h['item_cate'],0,item_cate_idx)
                    # print(item_cate.shape)
                    # sys.exit()
                    cate_loss = model.create_classify_loss(embedding_h['item_cate'], cate_label)
                    rate_loss = model.create_classify_loss(embedding_h['item_rate'], rate_label)
                elif args.pre_train_task == 2:
                    cate_loss = 0
                    rate_loss = model.create_classify_loss(embedding_h['item_rate'], rate_label)
                elif args.pre_train_task == 3:
                    cate_loss = model.create_classify_loss(embedding_h['item_cate'], cate_label)
                    rate_loss = 0
                elif args.pre_train_task == 4:
                    cate_loss = 0
                    rate_loss = 0
            if args.attrimask == 1:
                _, _, bpr_loss = model.create_bpr_loss(pos_g, neg_g, embedding_h, loss_type=args.loss)
            if 'xmrec' in args.dataset:
                if args.item_pretrain in [1, 2]:
                    neg_g_bT = construct_negative_item_graph(g, args.neg_samples, device, 'bT_idx')
                    bpr_loss_bT, mf_loss_bT, emb_loss_bT = model.create_item_bpr_loss(pos_g_bT, neg_g_bT, embedding_h)
                    # bpr_loss += bpr_loss_bT
                    if args.item_pretrain == 1 and args.dataset not in ['xmrec_cn']:
                        neg_g_cpr = construct_negative_item_graph(g, args.neg_samples, device, 'cpr_idx')
                        bpr_loss_cpr, _, _ = model.create_item_bpr_loss(pos_g_cpr, neg_g_cpr, embedding_h, 'cpr_idx')
                        # bpr_loss += bpr_loss_cpr
                elif args.item_pretrain == 3 and args.dataset not in ['xmrec_cn']:
                    neg_g_cpr = construct_negative_item_graph(g, args.neg_samples, device, 'cpr_idx')
                    bpr_loss_cpr, _, _ = model.create_item_bpr_loss(pos_g_cpr, neg_g_cpr, embedding_h, 'cpr_idx')
                    # bpr_loss += bpr_loss_cpr
            elif 'steam' in args.dataset:
                if args.user_pretrain in [1, 2]:
                    neg_g_age = construct_negative_user_graph(g, args.neg_samples, device, 'age')
                    # print(embedding_h.keys())
                    bpr_loss_age, mf_loss_age, emb_loss_age = model.create_user_bpr_loss(pos_g_age, neg_g_age,
                                                                                         embedding_h)
                    # bpr_loss += bpr_loss_age
                    if args.user_pretrain == 1:
                        neg_g_job = construct_negative_user_graph(g, args.neg_samples, device, 'job')
                        bpr_loss_job, mf_loss_job, emb_loss_job = model.create_user_bpr_loss(pos_g_job, neg_g_job,
                                                                                             embedding_h, 'job')
                        # bpr_loss += bpr_loss_job
                elif args.user_pretrain == 3:
                    neg_g_job = construct_negative_user_graph(g, args.neg_samples, device, 'job')
                    bpr_loss_job, mf_loss_job, emb_loss_job = model.create_user_bpr_loss(pos_g_job, neg_g_job,
                                                                                         embedding_h, 'job')
                    # bpr_loss += bpr_loss_job
            '''Uniformity loss for regularization'''
            # n_batch = 6
            # uniform_loss = 0
            # for idx in range(n_batch):
            #     uniform_loss += model.uniformity_loss_batched(embedding_h['item'], n_batch, idx)
            # ttl_loss = bpr_loss + cate_loss + rate_loss + 0.00001 * uniform_loss
            # ttl_loss = bpr_loss + cate_loss + rate_loss + bpr_loss_bT + bpr_loss_cpr + bpr_loss_age + bpr_loss_job + user_uniform_loss
            ttl_loss = 2 * (ui_loss_alpha * bpr_loss + (1 - ui_loss_alpha) * (
                    cate_loss + rate_loss + bpr_loss_bT + bpr_loss_cpr + bpr_loss_age + bpr_loss_job + user_uniform_loss + item_uniform_loss))
            optimizer.zero_grad()
            ttl_loss.backward()
            optimizer.step()
            if args.multitask_train == 1:
                if (epoch + 1) % (args.verbose * 10) != 0:
                    if args.verbose > 0 and epoch % args.verbose == 0:
                        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                            epoch, time() - t1, bpr_loss, mf_loss, emb_loss)
                        print(perf_str)
                    continue
                t2 = time()
                test_users = users_to_test
                ret, _, _ = test_cpp(test_users, embedding_h)
                t3 = time()
                loss_loger.append(bpr_loss)
                rec_loger.append(ret['recall'])
                ndcg_loger.append(ret['ndcg'])

                if args.verbose > 0:
                    perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                               'ndcg=[%.5f, %.5f]' % \
                               (epoch, t2 - t1, t3 - t2, bpr_loss, mf_loss, emb_loss, ret['recall'][0],
                                ret['recall'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                    print(perf_str)

                cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                            stopping_step, expected_order='acc',
                                                                            flag_step=args.flag_step)
                # if epoch == 99:
                if should_stop == True:
                    break
            else:
                if (epoch + 1) % (args.verbose * 10) != 0:
                    if args.verbose > 0 and epoch % (args.verbose * 10) == 0:
                        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f +%.5f + %.5f]' % (
                            epoch, time() - t_pre, ttl_loss, bpr_loss, cate_loss, rate_loss, user_uniform_loss)
                        print(perf_str)
                        # print('user-item loss = %.5f, embedding loss = %.5f.'%(mf_loss,emb_loss))
                    continue
                pre_train_best, stopping_step, should_stop = early_stopping(ttl_loss, pre_train_best,
                                                                            stopping_step, expected_order='dec',
                                                                            flag_step=args.flag_step)
                if should_stop == True:
                    # if epoch == 99:
                    print('Pre-train stopped.')
                    break
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # print(embedding_h['user'],embedding_h['item'])
    # sys.exit()
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    if args.pre_train == 0 or args.multitask_train == 0:
        if args.train_curve:
            '''1000/20 for cn, 500/10 for mx， 50/1 for steam'''
            args.flag_step = 500
            args.epoch = 500
        user_uniform_loss = 0
        item_uniform_loss = 0
        for epoch in range(args.epoch):
            t1 = time()
            neg_g = construct_negative_graph(g, args.neg_samples, device=device)
            if args.lightgcn == 1:
                embedding_h = model.lightgcn_forward(g)
            elif args.lightgcn == 2:
                embedding_h = {}
                embedding_h1 = model.lightgcn_forward(g)
                embedding_h2 = model(g, pre_train=False, linear_transform=True)
                embedding_h['user'] = embedding_h1['user'] + embedding_h2['user']
                embedding_h['item'] = embedding_h1['item'] + embedding_h2['item']
            elif args.lightgcn == 3:
                embedding_h = model.dhcf_forward(g)
            elif args.lightgcn == 4:
                embedding_h = model.ultragcn_forward()
                args.finetune_loss = 'ultragcn'
            else:
                embedding_h = model(g, pre_train=False)
            #print(embedding_h['user'].shape, embedding_h['item'].shape)
            #sys.exit()
            if args.inductive:
                bpr_loss, mf_loss, emb_loss = model.create_bpr_loss(pos_g, neg_g, embedding_h, users_non_induct,
                                                                    loss_type='bpr')
            else:
                bpr_loss, mf_loss, emb_loss = model.create_bpr_loss(pos_g, neg_g, embedding_h,
                                                                    loss_type=args.finetune_loss)
            if args.finetune_loss == 'auloss':
                user_uniform_loss = model.uniformity_loss(embedding_h['user'])
                item_uniform_loss = model.uniformity_loss(embedding_h['item'])
            bpr_loss = mf_loss + emb_loss + user_uniform_loss + item_uniform_loss

            if args.lightgcn == 2:
                bpr_loss += model.create_ssl_loss_user(embedding_h1['user'], embedding_h2['user'], 0.1)
            optimizer.zero_grad()
            bpr_loss.backward()
            optimizer.step()
            if args.train_curve != 1:
                if (epoch + 1) % (args.verbose * 10) != 0:
                    if args.verbose > 0 and epoch % args.verbose == 0:
                        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                            epoch, time() - t1, bpr_loss, mf_loss, emb_loss)
                        print(perf_str)
                    continue
            if args.dataset == 'xmrec_mx' and args.train_curve == 1:
                if epoch != 0:
                    if (epoch + 1) % (10) != 0:
                        if args.verbose > 0 and epoch % 1 == 0:
                            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                                epoch, time() - t1, bpr_loss, mf_loss, emb_loss)
                            print(perf_str)
                        continue
            if args.dataset == 'xmrec_cn' and args.train_curve == 1:
                if (epoch + 1) % (20) != 0:
                    if args.verbose > 0 and epoch % 2 == 0:
                        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                            epoch, time() - t1, bpr_loss, mf_loss, emb_loss)
                        print(perf_str)
                    continue
            t2 = time()
            if args.dataset == 'xmrec_mx':
                k=1
                test_users = users_to_test[k*16384-8192:(k+1)*16384-8192]
                if args.inductive:
                    test_users_transduct = test_users_transduct[k*16384-8192:(k+1)*16384-8192]
            else:
                test_users = users_to_test
                if args.inductive:
                    test_users_transduct = users_transduct
            # test_users = users_to_test
            '''Inference process'''
            if args.inductive:
                if args.lightgcn:
                    embedding_h_induct = model.lightgcn_forward(original_g)
                else:
                    embedding_h_induct = model(original_g, pre_train=False)
                    # embedding_h = model(original_g, pre_train=False)
            if args.fast_test:
                if args.inductive == 0:
                    ret, _, _ = test_cpp(test_users, embedding_h)
                else:
                    ret = {}
                    ret_inductive, _, _ = test_cpp(users_induct, embedding_h_induct, device=device)
                    ret_transduct, _, _ = test_cpp(test_users_transduct, embedding_h, device=device)
                    n_users_induct = len(users_induct)
                    n_users_transduct = len(test_users_transduct)
                    rec_loger_induct.append(ret_inductive['recall'])
                    ndcg_loger_induct.append(ret_inductive['ndcg'])
                    rec_loger_transduct.append(ret_transduct['recall'])
                    ndcg_loger_transduct.append(ret_transduct['ndcg'])

                    ret['recall'] = (ret_inductive['recall'] * n_users_induct + ret_transduct[
                        'recall'] * n_users_transduct) / (n_users_induct + n_users_transduct)
                    ret['ndcg'] = (ret_inductive['ndcg'] * n_users_induct + ret_transduct[
                        'ndcg'] * n_users_transduct) / (n_users_induct + n_users_transduct)
            else:
                ret, recall_dict, ndcg_dict = test(test_users, embedding_h, user_split=args.user_split)
            t3 = time()

            loss_loger.append(bpr_loss)
            rec_loger.append(ret['recall'])
            ndcg_loger.append(ret['ndcg'])

            # hit_loger.append(ret['hit_ratio'])

            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'ndcg=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, bpr_loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)
                if args.inductive:
                    perf_str_induct = ' recall_IN=[%.5f, %.5f], ndcg_IN=[%.5f, %.5f]' % \
                                      (ret_inductive['recall'][0], ret_inductive['recall'][-1],
                                       ret_inductive['ndcg'][0], ret_inductive['ndcg'][-1])
                    perf_str_transduct = ' recall_TR=[%.5f, %.5f], ndcg_TR=[%.5f, %.5f]' % \
                                         (ret_transduct['recall'][0], ret_transduct['recall'][-1],
                                          ret_transduct['ndcg'][0], ret_transduct['ndcg'][-1])
                    print(perf_str_induct)
                    print(perf_str_transduct)
            if args.inductive:
                ret_recall_0 = (ret['recall'][0] + ret_inductive['recall'][0]) / 2
            else:
                ret_recall_0 = ret['recall'][0]
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret_recall_0, cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=args.flag_step)

            # early stop
            if should_stop == True:
                break
    if args.train_curve:
        rec_10_data = json.dumps([i.tolist()[0] for i in rec_loger])
        ndcg_10_data = json.dumps([i.tolist()[0] for i in ndcg_loger])
        rec_20_data = json.dumps([i.tolist()[1] for i in rec_loger])
        ndcg_20_data = json.dumps([i.tolist()[1] for i in ndcg_loger])
        train_log_path = 'TrainLog/' + args.dataset
        if_pretrain = '/Pretrained/' if args.pre_train else '/No_pretrained/'
        if args.sgl:
            if_pretrain = '/SGL_pre/'
        elif args.gcc:
            if_pretrain = '/GCC/'
        elif args.attrimask:
            if_pretrain = '/Attrimask/'

        reg_10_path = train_log_path + if_pretrain + 'rec10_%d.json'%(args.random_seed)
        ndcg_10_path = train_log_path + if_pretrain +  'ndcg10_%d.json'%(args.random_seed)
        reg_20_path = train_log_path + if_pretrain + 'rec20_%d.json'%(args.random_seed)
        ndcg_20_path = train_log_path + if_pretrain +  'ndcg20_%d.json'%(args.random_seed)
        ensureDir(reg_10_path)
        ensureDir(ndcg_10_path)
        ensureDir(reg_20_path)
        ensureDir(ndcg_20_path)
        with open(reg_10_path, 'w') as reg_10_file:
            reg_10_file.write(rec_10_data)
        with open(reg_20_path, 'w') as reg_20_file:
            reg_20_file.write(rec_20_data)
        with open(ndcg_10_path, 'w') as ndcg_10_file:
            ndcg_10_file.write(ndcg_10_data)
        with open(ndcg_20_path, 'w') as ndcg_20_file:
            ndcg_20_file.write(ndcg_20_data)
    # print(embedding_h['user'],embedding_h['item'])
    # sys.exit()
    if args.save_flag == 1:
        group_emb = model.h_group_v1.cpu().detach().numpy()
        np.save(args.weights_path + args.dataset, group_emb)
        # user_emb = model.h_user_v1.cpu().detach().numpy()
        # torch.save(model.state_dict(), args.weights_path + args.model_name)
        print('save the weights in path: ', args.weights_path + args.dataset)
    if args.show_distance:
        a_score = model.alignment_score()
        u_uscore = model.uniformity_score_user()
        if 'steam' not in args.dataset:
            u_gscore = model.uniformity_score_group()
        else:
            u_gscore = model.uniformity_score_group_batch()
        print(a_score)
        print(u_uscore)
        print(u_gscore)
    recs = np.array(rec_loger)
    # pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    # hit = np.array(hit_loger)
    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)
    if args.inductive:
        recs_induct = np.array(rec_loger_induct)
        ndcgs_induct = np.array(ndcg_loger_induct)
        recs_transduct = np.array(rec_loger_transduct)
        ndcgs_transduct = np.array(ndcg_loger_transduct)
        # best_rec_0_induct = max(recs_induct[:, 0])
        # idx_induct = list(recs_induct[:, 0]).index(best_rec_0_induct)

    if args.show_distance:
        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s], ssl_metric=[%.5f, %.5f, %.5f]" % \
                     (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcgs[idx]]), a_score, u_uscore, u_gscore)
    else:
        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s]" % \
                     (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
        if args.inductive:
            final_perf += "\trecall_IN=[%s], ndcg_IN=[%s]" % \
                          ('\t'.join(['%.5f' % r for r in recs_induct[idx]]),
                           '\t'.join(['%.5f' % r for r in ndcgs_induct[idx]]))
            final_perf += "\trecall_TR=[%s], ndcg_TR=[%s]" % \
                          ('\t'.join(['%.5f' % r for r in recs_transduct[idx]]),
                           '\t'.join(['%.5f' % r for r in ndcgs_transduct[idx]]))
            final_perf += "\tInduct_ratio=" + str(args.induct_ratio)
    print(final_perf)

    save_path = './output/%s/%s.result' % (args.dataset, args.model_name)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'se=%d, beta_pool=%.2f, random_seed=%d, inductive=%d, inductive_ratio=%.2f, top_k=%s, layer_pool=%s, batch_size=%d, norm=%.1f, multitask_train=%d, \n'
        '\tpre_train=%d, loss=%s, finetune_loss=%s, hgcn=%d, lightgcn=%d, pre_train_task=%d, user_pretrain=%d, item_pretrain=%d, classify_as_edge=%d, \n'
        '\tlr=%.4f, pre_lr=%.4f, att_conv=%d, hgcn_mix=%s, regs=%s\n\t--%s\n'
        % (args.se, args.beta_pool, args.random_seed, args.inductive, args.induct_ratio, args.Ks, args.layer_pool, args.batch_size, args.norm_2, args.multitask_train,
           args.pre_train, args.loss, args.finetune_loss, args.hgcn, args.lightgcn, args.pre_train_task, args.user_pretrain, args.item_pretrain, args.classify_as_edge,
           args.lr, args.pre_lr, args.att_conv, args.hgcn_mix, args.regs, final_perf))
    f.close()
    if args.user_split == 1:
        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s]" % \
                     (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
        final_perf += '\n\trecalls: ' + convert_list_str(recall_split_loger[idx]) + '\n'
        final_perf += '\tndcgs:   ' + convert_list_str(ndcg_split_loger[idx]) + '\n'
        save_path_split = './output/%s/%s.result_case' % (args.dataset, args.model_name)
        ensureDir(save_path_split)
        f_split = open(save_path_split, 'a')
        f_split.write(
            'lr=%.4f, norm=%.1f, gat=%d, lightgcn=%d, hgcn=%d, pre_train_task=%d, hgcn_u_hyperedge=%d, user_hpedge_ig=%d, contrastive_learning=%d, ssl_reg=%.7f, ssl_temp=%.2f, reweight_type=%d, beta_group=%.2f, beta_item=%.2f,regs=%s\n\t%s\n'
            % (args.lr, args.norm_2, args.gat, args.lightgcn, args.hgcn, args.pre_train_task,
               args.hgcn_u_hyperedge,
               args.user_hpedge_ig, args.contrastive_learning, args.ssl_reg, args.ssl_temp, args.reweight_type,
               args.beta_group, args.beta_item,
               args.regs, final_perf))
        f_split.close()


if __name__ == '__main__':
    args = parse_args()
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True #使得网络相同输入下每次运行的输出固定
    dgl.seed(args.random_seed)

    args.model_name = 'AttriMask' #
    args.random_seed=132
    args.pre_lr=0.01
    args.lr=0.05
    args.regs='[0.7, 1e-4]'
    args.verbose=1
    args.layer_num=2
    args.se=0
    args.lightgcn=0 #LightGCN/DirectAU/SGL:1; HCCF:2; DHCF:3; UltraGCN:4; HGNN/GCC/AttriMask:0
    args.hgcn=1 #LightGCN/DirectAU/UltraGCN:0; HGNN/HCCF:1
    args.pre_train=1 #LightGCN/DirectAU/UltraGCN/HGNN/HCCF/DHCF:0; GCC/SGL/AttriMask:1
    args.finetune_loss='bpr' #LightGCN/HGNN/HCCF/DHCF:bpr; DirectAU:auloss; UltraGCN:ultragcn
    args.attrimask=1
    args.sgl=0
    args.gcc=0

    print(args)
    main(args)
