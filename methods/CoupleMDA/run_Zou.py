import sys
sys.path.append('../../')
import time
import argparse

import torch
import dgl
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn

from utils.data import load_Zou_data
from utils.tools import sort_ndarray, get_node_metapath_2_edges, find_nth_order_neighbors
from model.MAGNN_lp import MAGNN_lp
from utils.pytorchtools import datalodaer_gen, EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
from utils.com_reminder import send_email
from copy import deepcopy
import os

# Params
test_edge_type = 12  # 0-1
num_ntype = 8
lr = 2e-3
weight_decay = 1e-5

num_metapaths_list = [6, 6]
expected_metapaths = [
    [(0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 0), (0, 5, 0), (0, 7, 0)],
    [(1, 0, 1), (1, 2, 1), (1, 3, 1), (1, 5, 1), (1, 6, 1), (1, 7, 1)]
]

num_edge_type = 17
etypes_lists = [
    [[]]*num_metapaths_list[0],
    [[]]*num_metapaths_list[1]
]

use_masks = [[True, False, False, False, False, False],
             [True, False, False, False, False, False]]
no_masks = [[False] * 6, [False] * 6]

# link_11 param
num_link_type = 4+9
metapaths_01 = [
    (0, 2, 1), (0, 3, 1), (0, 5, 1), (0, 7, 1),
    (0, 1, 0, 1), (0, 2, 3, 1), (0, 2, 6, 1), (0, 3, 2, 1),
    (0, 3, 5, 1), (0, 4, 2, 1), (0, 4, 3, 1), (0, 4, 5, 1), (0, 5, 3, 1)
]
etypes_01 = [[]]*num_link_type

def run_model_Zou(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                     num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix, run_mode):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'using device {device}')
    adjlists_ua, edge_metapath_indices_list_ua, adjM, type_mask, dl, edge_metapath_indices_list_01 = load_Zou_data(save_postfix)
    # print(dl.links['meta'])
    # get_node_metapath_2_edges(edge_metapath_indices_list_12, [7147, 18683], save_file="link_node1_node2.csv")
    # find_nth_order_neighbors(adjM, 7024, 18754, n=1, output_file="neib_lastfm.csv")
    # exit(3)
    features_list = []
    in_dims = []
    if feats_type == 0:
        for i in range(num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse_coo_tensor(indices, values, torch.Size([dim, dim]), dtype=torch.float32, device=device))
    elif feats_type == 1:
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(device))
    elif feats_type == 2:
        raise NotImplementedError

    edge_types = [
        ('7', '(7,1)', '1'), ('1', '(1,7)', '7'),
        ('7', '(7,0)', '0'), ('0', '(0,7)', '7'),
        ('1', '(1,3)', '3'), ('3', '(3,1)', '1'),
        ('1', '(1,6)', '6'), ('6', '(6,1)', '1'),
        ('2', '(2,1)', '1'), ('1', '(1,2)', '2'),
        ('2', '(2,6)', '6'), ('6', '(6,2)', '2'),
        ('2', '(2,3)', '3'), ('3', '(3,2)', '2'),
        ('2', '(2,4)', '4'), ('4', '(4,2)', '2'),
        ('5', '(5,1)', '1'), ('1', '(1,5)', '5'),
        ('5', '(5,0)', '0'), ('0', '(0,5)', '5'),
        ('5', '(5,3)', '3'), ('3', '(3,5)', '5'),
        ('0', '(0,1)', '1'), ('1', '(1,0)', '0'),
        ('0', '(0,2)', '2'), ('2', '(2,0)', '0'),
        ('0', '(0,3)', '3'), ('3', '(3,0)', '0'),
        ('0', '(0,4)', '4'), ('4', '(4,0)', '0'),
        ('3', '(3,4)', '4'), ('4', '(4,3)', '3')
    ]
    srcs_dsts = [
        dl.links['data'][0].nonzero(), dl.links['data_trans'][0].nonzero(),
        dl.links['data'][1].nonzero(), dl.links['data_trans'][1].nonzero(),
        dl.links['data'][2].nonzero(), dl.links['data_trans'][2].nonzero(),
        dl.links['data'][3].nonzero(), dl.links['data_trans'][3].nonzero(),
        dl.links['data'][4].nonzero(), dl.links['data_trans'][4].nonzero(),
        dl.links['data'][5].nonzero(), dl.links['data_trans'][5].nonzero(),
        dl.links['data'][6].nonzero(), dl.links['data_trans'][6].nonzero(),
        dl.links['data'][7].nonzero(), dl.links['data_trans'][7].nonzero(),
        dl.links['data'][8].nonzero(), dl.links['data_trans'][8].nonzero(),
        dl.links['data'][9].nonzero(), dl.links['data_trans'][9].nonzero(),
        dl.links['data'][10].nonzero(), dl.links['data_trans'][10].nonzero(),
        dl.links['data'][11].nonzero(), dl.links['data_trans'][11].nonzero(),
        dl.links['data'][12].nonzero(), dl.links['data_trans'][12].nonzero(),
        dl.links['data'][13].nonzero(), dl.links['data_trans'][13].nonzero(),
        dl.links['data'][14].nonzero(), dl.links['data_trans'][14].nonzero(),
        dl.links['data'][15].nonzero(), dl.links['data_trans'][15].nonzero(),
    ]
    f_shift = [0] + [dl.nodes['shift'][i]+dl.nodes['count'][i] for i in range(num_ntype)]
    num_nodes_dict = {str(i): f_shift[-1] for i in range(num_ntype)}
    g_ori = dgl.heterograph({e: s for e, s, in zip(edge_types, srcs_dsts)}, num_nodes_dict=num_nodes_dict)

    g_selfloop = deepcopy(g_ori)
    for etype in edge_types:
        dst = int(etype[-1])
        self_loop_nodes = torch.arange(f_shift[dst], f_shift[dst+1])
        g_selfloop.add_edges(self_loop_nodes, self_loop_nodes, etype=etype[1])

    src, dst, etypes = [], [], []
    for i, s_d in enumerate(srcs_dsts):
        s, d = s_d
        src.extend(s.tolist())
        dst.extend(d.tolist())
        etypes.extend([i] * len(s))
    self_loop = list(range(len(type_mask)))
    src.extend(self_loop)
    dst.extend(self_loop)
    etypes.extend([len(srcs_dsts)] * len(type_mask))
    assert len(src) == len(dst) and len(src) == len(etypes)
    g_homo = dgl.graph((src, dst), num_nodes=len(type_mask))

    g = [g_ori.to(device), g_selfloop.to(device), g_homo.to(device), etypes]

    train_pos_user_artist = sort_ndarray(np.array(dl.train_pos[test_edge_type]).T)
    val_pos_user_artist = sort_ndarray(np.array(dl.valid_pos[test_edge_type]).T)
    train_neg_user_artist = sort_ndarray(np.array(dl.train_neg[test_edge_type]).T)
    val_neg_user_artist = sort_ndarray(np.array(dl.valid_neg[test_edge_type]).T)

    if run_mode == "train":
        dl_train = datalodaer_gen(train_pos_user_artist, train_neg_user_artist, batch_size=batch_size, shuffle=True)
    elif run_mode == "test":
        dl_val2 = datalodaer_gen(val_pos_user_artist, val_neg_user_artist, batch_size=batch_size * 2, shuffle=True)
    if run_mode != "discover":
        dl_val = datalodaer_gen(val_pos_user_artist, val_neg_user_artist, batch_size=batch_size*2, shuffle=False)

    datas = (g, adjlists_ua, edge_metapath_indices_list_ua, type_mask, features_list, adjM,
             None, edge_metapath_indices_list_01, dl)
    param = (lr, weight_decay, num_epochs, device, neighbor_samples, use_masks, no_masks, logfile_path, p_data_path)
    early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=checkpoint_path)
    net = MAGNN_lp(num_metapaths_list, num_edge_type, etypes_lists, etypes_01,
                   in_dims, hidden_dim, num_heads, attn_vec_dim, edge_types=edge_types, f_shift=f_shift,
                   rnn_type=rnn_type, l_enco_type='linear', dropout_rate=0.0, num_link_type=num_link_type,
                   use_self_attn=True, meta_mode='just_link', inner_attn=True, self_super_meta=False)

    net.to(device)

    if run_mode == "train":
        net.fit(dl_train, dl_val, datas, param, early_stopping)
        net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        net.test(dl, batch_size * 2, datas, param, test_edge_type)
    elif run_mode == "test":
        net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        val_loss, result = net.evaluate(dl_val, datas, param)
        print('val1 result')
        print(result)

        val_loss, result = net.evaluate(dl_val2, datas, param)
        print('val2 result')
        print(result)

        net.test(dl, batch_size*2, datas, param, test_edge_type)
    else:
        net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        net.discover_specific_links(dl, batch_size*2, datas, param, test_edge_type=test_edge_type,
                                   link_discover_opath="../../data/Zou/link_discover_TNBC_id.csv",
                                   output_file=f"pred_data/pred_Zou_miRNAs_TNBC.csv")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the recommendation dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector;' +
                         '2 - Node feature themselves. Default is 0 for Zou.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='linear', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=30, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=256, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=500, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--dataset', default='Zou', help='Postfix for the saved model and result. Default is Zou.')
    ap.add_argument('--run-mode', default='train', help='run_mode in ["train", "test", "discover"]')

    args = ap.parse_args()

    checkpoint_path = f'checkpoint/checkpoint_{args.dataset}_2.pt'
    logfile_path = f"output/output_{args.dataset}_2.txt"
    p_data_path = f"pred_data/p_data_{args.dataset}_2.csv"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    os.makedirs(os.path.dirname(p_data_path), exist_ok=True)

    assert args.run_mode in ["train", "test", "discover"]

    if True:
        run_model_Zou(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, args.epoch,
                         args.patience, args.batch_size, args.samples, args.repeat, args.dataset, args.run_mode)
    else:
        try:
            run_model_Zou(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, args.epoch,
                             args.patience, args.batch_size, args.samples, args.repeat, args.dataset)
            send_email("success")    # 程序运行完成提醒
        except Exception as e:
            send_email("error")
