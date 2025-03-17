import copy
import math
import os.path
import random
import time

import networkx as nx
import dgl
import scipy.sparse as sp
import numpy as np
import scipy
import pickle
import torch
import sys


def get_meta_dict_01(dl, meta: tuple) -> dict:
    # meta: (src, dst) or (src, edge, dst) type
    if not dl.nonzero:
        print("getting nonzero, please wait...")
        t0 = time.time()
        dl.get_nonzero()
        print(f"run get nonzero time = {time.time() - t0:.2f}s")
    if not isinstance(meta, tuple) or len(meta) not in [2, 3]:
        raise TypeError
    edge_type = dl.get_edge_type(meta)
    start_node_type = meta[0]
    meta_dict = {}

    trav = range(dl.nodes['shift'][start_node_type],
                 dl.nodes['shift'][start_node_type] + dl.nodes['count'][start_node_type])
    for i in trav:
        meta_dict[i] = []
        dl.dfs([i], [edge_type], meta_dict)
    return meta_dict


def cat_metapath(dl, meta_01: dict, meta_12: dict, start_node_type=0) -> dict:
    # 拼接：拼接meta01和meta12得到meta012
    shift = dl.nodes['shift'][start_node_type]
    meta_012 = {}
    for i in range(shift, shift + dl.nodes['count'][start_node_type]):
        meta_012[i] = []
        for beg in meta_01[i]:
            for end in meta_12[beg[-1]]:
                meta_012[i].append(beg + end[1:])
    return meta_012


def sample_cat_metapath(dl, meta_01: dict, meta_12: dict, start_node_type=0, sample_rate=0.45) -> dict:
    # 采样拼接：由meta01和meta12采样拼接得meta012
    meta_012 = {}
    for i in range(dl.nodes['shift'][start_node_type],
                   dl.nodes['shift'][start_node_type] + dl.nodes['count'][start_node_type]):
        meta_012[i] = []
        sample_01_edges = random.sample(meta_01[i], math.ceil(len(meta_01[i]) * sample_rate))
        for beg in sample_01_edges:
            sample_12_edges = random.sample(meta_12[beg[-1]], math.ceil(len(meta_12[beg[-1]]) * sample_rate))
            for end in sample_12_edges:
                meta_012[i].append(beg + end[1:])
    return meta_012


def sym_metapath(dl, meta_01: dict, end_node_type=1) -> dict:
    # 对称：由meta01对称得到meta10
    shift = dl.nodes['shift'][end_node_type]
    meta_10 = {}
    for i in range(shift, shift + dl.nodes['count'][end_node_type]):
        meta_10[i] = []
    for paths in meta_01.values():
        for path in paths:
            meta_10[path[-1]].append(path[::-1])
    return meta_10


def get_adjlist_idx(meta010: dict, meta_len) -> (dict, dict):
    adjlist00, idx00 = {}, {}
    for k in meta010:
        idx00[k] = np.array(sorted([tuple(reversed(i)) for i in meta010[k]]), dtype=np.int32).reshape([-1, meta_len])
        adjlist00[k] = idx00[k][:, 0].tolist()
    return adjlist00, idx00


def get_adjlist_idx_condition(meta010: dict, meta_len) -> (dict, dict):
    adjlist00, idx00 = {}, {}
    for k in meta010:
        idx00[k] = np.array(sorted([tuple(reversed(i)) for i in meta010[k]]), dtype=np.int32).reshape([-1, meta_len])
        data = idx00[k]
        condition = (data[:, 0] == data[:, 2]) | (data[:, 1] == data[:, 3])
        idx00[k] = idx00[k][~condition]
        adjlist00[k] = idx00[k][:, 0].tolist()
    return adjlist00, idx00


def get_adjlist_idx_condition3(meta010: dict, meta_len) -> (dict, dict):
    adjlist00, idx00 = {}, {}
    for k in meta010:
        idx00[k] = np.array(sorted([tuple(reversed(i)) for i in meta010[k]]), dtype=np.int32).reshape([-1, meta_len])
        data = idx00[k]
        condition = (data[:, 0] == data[:, 2])
        idx00[k] = idx00[k][~condition]
        adjlist00[k] = idx00[k][:, 0].tolist()
    return adjlist00, idx00


def just_test(dict_metapath: dict, dl, sample_size=100):
    keys = list(dict_metapath.keys())
    for i in range(sample_size):
        key = np.random.choice(keys)
        array = dict_metapath[key]
        if array.shape[0] == 0:
            continue
        row_index = np.random.randint(0, array.shape[0])
        sr = array[row_index]

        # 00
        # print("{:.0f}, {:.0f}".format(
        #     dl.links["data"][0][sr[0], sr[1]],
        #     dl.links["data_trans"][0][sr[1], sr[2]]))

        # 01
        print("{:.0f}, {:.0f}, {:.0f}, {:.0f}".format(
            dl.links["data"][0][sr[0], sr[1]],
            dl.links["data"][2][sr[1], sr[2]],
            dl.links["data_trans"][2][sr[2], sr[3]],
            dl.links["data_trans"][0][sr[3], sr[4]]))

        # 02
        # print("key:{} edge:({}, {}) {:.0f}".format(key, sr[0], sr[1], dl.links["data"][1][sr[0], sr[1]]))

        # # 10
        # print("{:.0f}, {:.0f}".format(
        #     dl.links["data_trans"][0][sr[0], sr[1]],
        #     dl.links["data"][0][sr[1], sr[2]]))

        # # 11
        # print("{:.0f}, {:.0f}".format(
        #     dl.links["data"][2][sr[0], sr[1]],
        #     dl.links["data_trans"][2][sr[1], sr[2]]))

        # 12
        # print("{:.0f}, {:.0f}, {:.0f}".format(
        #     dl.links["data_trans"][0][sr[0], sr[1]],
        #     dl.links["data"][1][sr[1], sr[2]],
        #     dl.links["data"][0][sr[2], sr[3]]))


def load_LastFM_data(dataset='LastFM'):
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/'+dataset)
    if os.path.exists(f"../../data/{dataset}/adj_and_idx1.pkl"):
    # if False:
        with open(f"../../data/{dataset}/adj_and_idx1.pkl", 'rb') as file:
            adjlists_ua, edge_metapath_indices_list_ua = pickle.load(file)
        print(f"loading data/{dataset}/adj_and_idx1.pkl")
        # adjlists_ua, edge_metapath_indices_list_ua = None, None

        with open(f"../../data/{dataset}/adj_and_idx2.pkl", 'rb') as file:
            adjlists_01, edge_metapath_indices_list_01 = pickle.load(file)
        print(f"loading data/{dataset}/adj_and_idx2.pkl")
    else:
        meta_01 = get_meta_dict_01(dl, (0, 1))
        meta_00 = get_meta_dict_01(dl, (0, 0))
        meta_12 = get_meta_dict_01(dl, (1, 2))
        print("get_meta_dict_12 finish")
        meta_10 = sym_metapath(dl, meta_01, end_node_type=1)
        meta_21 = sym_metapath(dl, meta_12, end_node_type=2)
        meta_010 = cat_metapath(dl, meta_01, meta_10, start_node_type=0)
        meta_121 = cat_metapath(dl, meta_12, meta_21, start_node_type=1)
        meta_012 = sample_cat_metapath(dl, meta_01, meta_12, start_node_type=0, sample_rate=0.45)
        meta_210 = sym_metapath(dl, meta_012, end_node_type=2)
        print("get_meta_dict_210 finish")
        meta_01210 = cat_metapath(dl, meta_012, meta_210, start_node_type=0)
        print("get_meta_dict_01210 finish")
        meta_101 = cat_metapath(dl, meta_10, meta_01, start_node_type=1)
        meta_001 = cat_metapath(dl, meta_00, meta_01, start_node_type=0)
        print("get_meta_dict_001 finish")
        meta_1001 = cat_metapath(dl, meta_10, meta_001, start_node_type=1)
        meta_0101 = cat_metapath(dl, meta_010, meta_01, start_node_type=0)
        print("get_meta_dict_0101 finish")
        meta_0121 = cat_metapath(dl, meta_01, meta_121, start_node_type=0)
        meta_0001 = cat_metapath(dl, meta_00, meta_001, start_node_type=0)
        print("get_meta_dict_0001 finish")
        del meta_01, meta_12, meta_10, meta_21, meta_012, meta_210

        adjlist00, idx00 = get_adjlist_idx(meta_010, meta_len=3)
        del meta_010
        adjlist01, idx01 = get_adjlist_idx(meta_01210, meta_len=5)
        del meta_01210
        adjlist02, idx02 = get_adjlist_idx(meta_00, meta_len=2)
        adjlist10, idx10 = get_adjlist_idx(meta_101, meta_len=3)
        del meta_00, meta_101
        adjlist11, idx11 = get_adjlist_idx(meta_121, meta_len=3)
        adjlist12, idx12 = get_adjlist_idx(meta_1001, meta_len=4)
        del meta_121, meta_1001
        print("get_adjlist12 finish")

        adjlist001, idx001 = get_adjlist_idx(meta_001, meta_len=3)
        adjlist0101, idx0101 = get_adjlist_idx_condition(meta_0101, meta_len=4)     # !!!
        del meta_001, meta_0101
        adjlist0121, idx0121 = get_adjlist_idx_condition(meta_0121, meta_len=4)
        adjlist0001, idx0001 = get_adjlist_idx_condition(meta_0001, meta_len=4)
        del meta_0121, meta_0001
        print("get_adjlist0001 finish")

        adjlists_ua, edge_metapath_indices_list_ua = [[adjlist00, adjlist01, adjlist02], [adjlist10, adjlist11, adjlist12]], \
                                                     [[idx00, idx01, idx02], [idx10, idx11, idx12]]
        adjlists_01, edge_metapath_indices_list_01 = [adjlist001, adjlist0101, adjlist0121, adjlist0001], \
                                                     [idx001, idx0101, idx0121, idx0001]
        with open(f"../../data/{dataset}/adj_and_idx1.pkl", 'wb') as file:
            pickle.dump((adjlists_ua, edge_metapath_indices_list_ua), file)
        with open(f"../../data/{dataset}/adj_and_idx2.pkl", 'wb') as file:
            pickle.dump((adjlists_01, edge_metapath_indices_list_01), file)

    adjM = sum(dl.links['data'].values())
    adjM = adjM + adjM.T
    adjM.data[:] = 1
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i

    return adjlists_ua, edge_metapath_indices_list_ua, adjM, type_mask, dl, adjlists_01, edge_metapath_indices_list_01


def load_PubMed_data(dataset='PubMed'):
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/'+dataset)
    if os.path.exists(f"../../data/{dataset}/adj_and_idx1.pkl"):
    # if False:
        with open(f"../../data/{dataset}/adj_and_idx1.pkl", 'rb') as file:
            adjlists_ua, edge_metapath_indices_list_ua = pickle.load(file)
        print(f"loading data/{dataset}/adj_and_idx1.pkl")
        # adjlists_ua, edge_metapath_indices_list_ua = None, None

        with open(f"../../data/{dataset}/adj2.pkl", 'rb') as file:
            edge_metapath_indices_list_11 = pickle.load(file)
        print(f"loading data/{dataset}/adj2.pkl")
    else:
        meta_00 = get_meta_dict_01(dl, (0, 0))
        meta_01 = get_meta_dict_01(dl, (0, 1))
        meta_10 = sym_metapath(dl, meta_01, end_node_type=1)
        meta_11 = get_meta_dict_01(dl, (1, 1))
        meta_12 = get_meta_dict_01(dl, (1, 2))
        meta_21 = sym_metapath(dl, meta_12, end_node_type=2)
        meta_13 = get_meta_dict_01(dl, (1, 3))
        meta_31 = sym_metapath(dl, meta_13, end_node_type=3)
        meta_20 = get_meta_dict_01(dl, (2, 0))
        meta_22 = get_meta_dict_01(dl, (2, 2))
        meta_30 = get_meta_dict_01(dl, (3, 0))
        meta_32 = get_meta_dict_01(dl, (3, 2))
        meta_33 = get_meta_dict_01(dl, (3, 3))
        print("get_meta_dict_xx finish")

        meta_101 = cat_metapath(dl, meta_10, meta_01, start_node_type=1)
        meta_111 = cat_metapath(dl, meta_11, meta_11, start_node_type=1)
        meta_121 = cat_metapath(dl, meta_12, meta_21, start_node_type=1)
        meta_131 = cat_metapath(dl, meta_13, meta_31, start_node_type=1)
        print("get_meta_dict_1x1 finish")

        sr = 1
        meta_100s = sample_cat_metapath(dl, meta_10, meta_00, start_node_type=1, sample_rate=sr)
        meta_1001s = sample_cat_metapath(dl, meta_100s, meta_01, start_node_type=1, sample_rate=sr)
        del meta_00, meta_10, meta_100s
        meta_1101s = sample_cat_metapath(dl, meta_11, meta_101, start_node_type=1, sample_rate=sr)
        meta_1111s = sample_cat_metapath(dl, meta_11, meta_111, start_node_type=1, sample_rate=sr)
        print("get_meta_dict_11x1s finish")

        meta_120s = sample_cat_metapath(dl, meta_12, meta_20, start_node_type=1, sample_rate=sr)
        meta_1201s = sample_cat_metapath(dl, meta_120s, meta_01, start_node_type=1, sample_rate=sr)
        meta_1211s = sample_cat_metapath(dl, meta_121, meta_11, start_node_type=1, sample_rate=sr)
        meta_122s = sample_cat_metapath(dl, meta_12, meta_22, start_node_type=1, sample_rate=sr)
        meta_1221s = sample_cat_metapath(dl, meta_122s, meta_21, start_node_type=1, sample_rate=sr)
        del meta_20, meta_12, meta_22, meta_122s, meta_120s
        print("get_meta_dict_12x1s finish")

        meta_130s = sample_cat_metapath(dl, meta_13, meta_30, start_node_type=1, sample_rate=sr)
        del meta_30
        meta_1301s = sample_cat_metapath(dl, meta_130s, meta_01, start_node_type=1, sample_rate=sr)
        del meta_01, meta_130s
        meta_131s = sample_cat_metapath(dl, meta_13, meta_31, start_node_type=1, sample_rate=sr)
        meta_1311s = sample_cat_metapath(dl, meta_131s, meta_11, start_node_type=1, sample_rate=sr)
        meta_132s = sample_cat_metapath(dl, meta_13, meta_32, start_node_type=1, sample_rate=sr)
        del meta_32, meta_131s,
        meta_1321s = sample_cat_metapath(dl, meta_132s, meta_21, start_node_type=1, sample_rate=sr)
        del meta_21, meta_132s
        meta_133s = sample_cat_metapath(dl, meta_13, meta_33, start_node_type=1, sample_rate=sr)
        del meta_13, meta_33
        meta_1331s = sample_cat_metapath(dl, meta_133s, meta_31, start_node_type=1, sample_rate=sr)
        del meta_31, meta_133s
        print("get_meta_dict_13x1s finish")

        adjlist11, idx11 = get_adjlist_idx(meta_11, meta_len=2)
        adjlist101, idx101 = get_adjlist_idx(meta_101, meta_len=3)
        _, idx111 = get_adjlist_idx(meta_111, meta_len=3)
        adjlist121, idx121 = get_adjlist_idx(meta_121, meta_len=3)
        adjlist131, idx131 = get_adjlist_idx(meta_131, meta_len=3)
        del meta_11, meta_101, meta_111, meta_121, meta_131
        print("get_adjlist1x1 finish")

        _, idx1001 = get_adjlist_idx(meta_1001s, meta_len=4)
        _, idx1101 = get_adjlist_idx_condition(meta_1101s, meta_len=4)  # !!!
        _, idx1111 = get_adjlist_idx_condition(meta_1111s, meta_len=4)
        del meta_1001s, meta_1101s, meta_1111s
        print("get_adjlist11x1 finish")

        _, idx1201 = get_adjlist_idx(meta_1201s, meta_len=4)
        _, idx1211 = get_adjlist_idx_condition(meta_1211s, meta_len=4)
        _, idx1221 = get_adjlist_idx(meta_1221s, meta_len=4)
        del meta_1201s, meta_1211s, meta_1221s
        print("get_adjlist12x1 finish")

        _, idx1301 = get_adjlist_idx(meta_1301s, meta_len=4)
        _, idx1311 = get_adjlist_idx_condition(meta_1311s, meta_len=4)
        _, idx1321 = get_adjlist_idx(meta_1321s, meta_len=4)
        _, idx1331 = get_adjlist_idx(meta_1331s, meta_len=4)
        del meta_1301s, meta_1311s, meta_1321s, meta_1331s
        print("get_adjlist13x1 finish")

        adjlists_ua, edge_metapath_indices_list_ua = [[adjlist11, adjlist101, adjlist121, adjlist131]], \
                                                     [[idx11, idx101, idx121, idx131]]
        edge_metapath_indices_list_11 = [idx101, idx111, idx121, idx131,
                       idx1001, idx1101, idx1111,
                       idx1201, idx1211, idx1221,
                       idx1301, idx1311, idx1321, idx1331]
        with open(f"../../data/{dataset}/adj_and_idx1.pkl", 'wb') as file:
            pickle.dump((adjlists_ua, edge_metapath_indices_list_ua), file)
        with open(f"../../data/{dataset}/adj2.pkl", 'wb') as file:
            pickle.dump(edge_metapath_indices_list_11, file)

    adjM = sum(dl.links['data'].values())
    adjM = adjM + adjM.T
    adjM.data[:] = 1
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i

    return adjlists_ua, edge_metapath_indices_list_ua, adjM, type_mask, dl, edge_metapath_indices_list_11


def load_DBLP_data(dataset='DBLP'):
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/'+dataset)
    if os.path.exists(f"../../data/{dataset}/adj_and_idx1.pkl"):
    # if False:
        with open(f"../../data/{dataset}/adj_and_idx1.pkl", 'rb') as file:
            adjlists_ua, edge_metapath_indices_list_ua = pickle.load(file)
        print(f"loading data/{dataset}/adj_and_idx1.pkl")
        # adjlists_ua, edge_metapath_indices_list_ua = None, None

        with open(f"../../data/{dataset}/adj2.pkl", 'rb') as file:
            edge_metapath_indices_list_12 = pickle.load(file)
        print(f"loading data/{dataset}/adj2.pkl")
    else:
        meta_01 = get_meta_dict_01(dl, (0, 1))
        meta_10 = sym_metapath(dl, meta_01, end_node_type=1)
        meta_12 = get_meta_dict_01(dl, (1, 2))
        meta_21 = sym_metapath(dl, meta_12, end_node_type=2)
        meta_13 = get_meta_dict_01(dl, (1, 3))
        meta_31 = sym_metapath(dl, meta_13, end_node_type=3)
        print("get_meta_dict_xx finish")
        print(f"meta_12 size={get_total_size(meta_12)}")
        print(f"meta_21 size={get_total_size(meta_21)}")

        meta_101 = cat_metapath(dl, meta_10, meta_01, start_node_type=1)
        meta_121 = cat_metapath(dl, meta_12, meta_21, start_node_type=1)
        meta_131 = cat_metapath(dl, meta_13, meta_31, start_node_type=1)
        meta_212 = cat_metapath(dl, meta_21, meta_12, start_node_type=2)
        print("get_meta_dict_xxx finish")

        meta_210s = cat_metapath(dl, meta_21, meta_10, start_node_type=2)
        meta_012s = sym_metapath(dl, meta_210s, end_node_type=0)
        meta_21012s = cat_metapath(dl, meta_210s, meta_012s, start_node_type=2)
        print(f"meta_210s size={get_total_size(meta_210s)}")
        print(f"meta_012s size={get_total_size(meta_012s)}")
        print(f"meta_21012s size={get_total_size(meta_21012s)}")
        del meta_01, meta_10, meta_210s, meta_012s
        print("get_meta_dict_21012 finish")

        sr = 0.5
        meta_213s = sample_cat_metapath(dl, meta_21, meta_13, start_node_type=2, sample_rate=sr)
        meta_312s = sym_metapath(dl, meta_213s, end_node_type=3)
        meta_21312s = sample_cat_metapath(dl, meta_213s, meta_312s, start_node_type=2, sample_rate=sr)
        print(f"meta_213s size={get_total_size(meta_213s)}")
        print(f"meta_312s size={get_total_size(meta_312s)}")
        print(f"meta_21312s size={get_total_size(meta_21312s)}")
        del meta_21, meta_13, meta_213s, meta_312s
        print("get_meta_dict_21312 finish")

        meta_1012 = cat_metapath(dl, meta_101, meta_12, start_node_type=1)
        meta_1312 = cat_metapath(dl, meta_131, meta_12, start_node_type=1)
        meta_1212 = cat_metapath(dl, meta_121, meta_12, start_node_type=1)
        del meta_12
        print("get_meta_dict_1x12 finish")

        adjlist101, idx101 = get_adjlist_idx(meta_101, meta_len=3)
        adjlist121, idx121 = get_adjlist_idx(meta_121, meta_len=3)
        adjlist131, idx131 = get_adjlist_idx(meta_131, meta_len=3)
        del meta_101, meta_121, meta_131
        print("get_adjlist_1x1 finish")

        adjlist212, idx212 = get_adjlist_idx(meta_212, meta_len=3)
        adjlist21012, idx21012 = get_adjlist_idx(meta_21012s, meta_len=5)
        adjlist21312, idx21312 = get_adjlist_idx(meta_21312s, meta_len=5)
        del meta_212, meta_21012s, meta_21312s
        print("get_adjlist_2x2 finish")

        _, idx1012 = get_adjlist_idx_condition(meta_1012, meta_len=4)  # !!!
        _, idx1212 = get_adjlist_idx_condition(meta_1212, meta_len=4)
        _, idx1312 = get_adjlist_idx_condition(meta_1312, meta_len=4)
        del meta_1012, meta_1212, meta_1312
        print("get_adjlist_1x12 finish")

        adjlists_ua, edge_metapath_indices_list_ua = [[adjlist101, adjlist121, adjlist131], [adjlist212, adjlist21012, adjlist21312]], \
                                                     [[idx101, idx121, idx131], [idx212, idx21012, idx21312]]
        edge_metapath_indices_list_12 = [
            idx1012, idx1212, idx1312
        ]
        with open(f"../../data/{dataset}/adj_and_idx1.pkl", 'wb') as file:
            pickle.dump((adjlists_ua, edge_metapath_indices_list_ua), file)
        with open(f"../../data/{dataset}/adj2.pkl", 'wb') as file:
            pickle.dump(edge_metapath_indices_list_12, file)

    adjM = sum(dl.links['data'].values())
    adjM = adjM + adjM.T
    adjM.data[:] = 1
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i

    return adjlists_ua, edge_metapath_indices_list_ua, adjM, type_mask, dl, edge_metapath_indices_list_12


def load_amazon_data(dataset='amazon'):
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/'+dataset)
    if os.path.exists(f"../../data/{dataset}/adj_and_idx1.pkl"):
    # if False:
        with open(f"../../data/{dataset}/adj_and_idx1.pkl", 'rb') as file:
            adjlists_ua, edge_metapath_indices_list_ua = pickle.load(file)
        print(f"loading data/{dataset}/adj_and_idx1.pkl")
        # adjlists_ua, edge_metapath_indices_list_ua = None, None

        with open(f"../../data/{dataset}/adj2.pkl", 'rb') as file:
            edge_metapath_indices_list_00 = pickle.load(file)
        print(f"loading data/{dataset}/adj2.pkl")
    else:
        meta_0A0 = get_meta_dict_01(dl, (0, 0, 0))
        meta_0B0 = get_meta_dict_01(dl, (0, 1, 0))
        print("get_meta_dict_x finish")

        meta_0AA0 = cat_metapath(dl, meta_0A0, meta_0A0, start_node_type=0)
        meta_0AB0 = cat_metapath(dl, meta_0A0, meta_0B0, start_node_type=0)
        meta_0BA0 = sym_metapath(dl, meta_0AB0, end_node_type=0)
        meta_0BB0 = cat_metapath(dl, meta_0B0, meta_0B0, start_node_type=0)
        print("get_meta_dict_xx finish")

        sr = 0.8
        meta_0AA0s = sample_cat_metapath(dl, meta_0A0, meta_0A0, start_node_type=0, sample_rate=sr)
        meta_0AB0s = sample_cat_metapath(dl, meta_0A0, meta_0B0, start_node_type=0, sample_rate=sr)
        meta_0BB0s = sample_cat_metapath(dl, meta_0B0, meta_0B0, start_node_type=0, sample_rate=sr)
        print("get_meta_dict_xxs finish")

        meta_0AAA0 = sample_cat_metapath(dl, meta_0AA0s, meta_0A0, start_node_type=0, sample_rate=sr)
        meta_0AAB0 = sample_cat_metapath(dl, meta_0AA0s, meta_0B0, start_node_type=0, sample_rate=sr)
        print("get_meta_dict_AAx finish")
        del meta_0AA0s
        meta_0ABA0 = sample_cat_metapath(dl, meta_0AB0s, meta_0A0, start_node_type=0, sample_rate=sr)
        meta_0ABB0 = sample_cat_metapath(dl, meta_0AB0s, meta_0B0, start_node_type=0, sample_rate=sr)
        print("get_meta_dict_ABx finish")
        meta_0BAA0 = sym_metapath(dl, meta_0AAB0, end_node_type=0)
        meta_0BAB0 = sample_cat_metapath(dl, meta_0B0, meta_0AB0s, start_node_type=0, sample_rate=sr)
        print("get_meta_dict_BAx finish")
        meta_0BBA0 = sym_metapath(dl, meta_0ABB0, end_node_type=0)
        del meta_0AB0s
        meta_0BBB0 = sample_cat_metapath(dl, meta_0BB0s, meta_0B0, start_node_type=0, sample_rate=sr)
        del meta_0BB0s
        print("get_meta_dict_BBx finish")

        adjlist0A0, idx0A0 = get_adjlist_idx(meta_0A0, meta_len=2)
        adjlist0B0, idx0B0 = get_adjlist_idx(meta_0B0, meta_len=2)
        del meta_0A0, meta_0B0
        print("get_adjlist_idx_x finish")

        adjlist0AA0, idx0AA0 = get_adjlist_idx_condition3(meta_0AA0, meta_len=3)
        adjlist0AB0, idx0AB0 = get_adjlist_idx_condition3(meta_0AB0, meta_len=3)
        adjlist0BA0, idx0BA0 = get_adjlist_idx_condition3(meta_0BA0, meta_len=3)
        adjlist0BB0, idx0BB0 = get_adjlist_idx_condition3(meta_0BB0, meta_len=3)
        del meta_0AA0, meta_0AB0, meta_0BA0, meta_0BB0
        print("get_adjlist_idx_xx finish")

        adjlist0AAA0, idx0AAA0 = get_adjlist_idx_condition(meta_0AAA0, meta_len=4)
        adjlist0AAB0, idx0AAB0 = get_adjlist_idx_condition(meta_0AAB0, meta_len=4)
        adjlist0ABA0, idx0ABA0 = get_adjlist_idx_condition(meta_0ABA0, meta_len=4)
        adjlist0ABB0, idx0ABB0 = get_adjlist_idx_condition(meta_0ABB0, meta_len=4)
        del meta_0AAA0, meta_0AAB0, meta_0ABA0, meta_0ABB0
        print("get_adjlist_idx_Axx finish")
        adjlist0BAA0, idx0BAA0 = get_adjlist_idx_condition(meta_0BAA0, meta_len=4)
        adjlist0BAB0, idx0BAB0 = get_adjlist_idx_condition(meta_0BAB0, meta_len=4)
        adjlist0BBA0, idx0BBA0 = get_adjlist_idx_condition(meta_0BBA0, meta_len=4)
        adjlist0BBB0, idx0BBB0 = get_adjlist_idx_condition(meta_0BBB0, meta_len=4)
        del meta_0BAA0, meta_0BAB0, meta_0BBA0, meta_0BBB0
        print("get_adjlist_idx_Bxx finish")

        adjlists_ua = [[adjlist0A0, adjlist0B0, adjlist0AA0, adjlist0AB0, adjlist0BA0, adjlist0BB0,
                        adjlist0AAA0, adjlist0AAB0, adjlist0ABA0, adjlist0ABB0,
                        adjlist0BAA0, adjlist0BAB0, adjlist0BBA0, adjlist0BBB0]]
        edge_metapath_indices_list_ua = [[idx0A0, idx0B0, idx0AA0, idx0AB0, idx0BA0, idx0BB0,
                                          idx0AAA0, idx0AAB0, idx0ABA0, idx0ABB0,
                                          idx0BAA0, idx0BAB0, idx0BBA0, idx0BBB0]]
        edge_metapath_indices_list_00 = [idx0AA0, idx0AB0, idx0BB0,
                                         idx0AAA0, idx0AAB0, idx0ABA0, idx0ABB0, idx0BAB0, idx0BBB0]
        with open(f"../../data/{dataset}/adj_and_idx1.pkl", 'wb') as file:
            pickle.dump((adjlists_ua, edge_metapath_indices_list_ua), file)
        with open(f"../../data/{dataset}/adj2.pkl", 'wb') as file:
            pickle.dump(edge_metapath_indices_list_00, file)

    adjM = sum(dl.links['data'].values())
    adjM = adjM + adjM.T
    adjM.data[:] = 1
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i

    return adjlists_ua, edge_metapath_indices_list_ua, adjM, type_mask, dl, edge_metapath_indices_list_00


def load_Yamanishi_data(dataset='Yamanishi/Enzyme'):
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/'+dataset, use_unknown_link=True)
    if os.path.exists(f"../../data/{dataset}/adj_and_idx1.pkl"):
    # if False:
        with open(f"../../data/{dataset}/adj_and_idx1.pkl", 'rb') as file:
            adjlists_ua, edge_metapath_indices_list_ua = pickle.load(file)
        print(f"loading data/{dataset}/adj_and_idx1.pkl")
        # adjlists_ua, edge_metapath_indices_list_ua = None, None

        with open(f"../../data/{dataset}/adj2.pkl", 'rb') as file:
            edge_metapath_indices_list_01 = pickle.load(file)
        print(f"loading data/{dataset}/adj2.pkl")
    else:
        meta_00 = get_meta_dict_01(dl, (0, 0))
        meta_01 = get_meta_dict_01(dl, (0, 1))
        meta_10 = sym_metapath(dl, meta_01, end_node_type=1)
        meta_11 = get_meta_dict_01(dl, (1, 1))
        print("get_meta_dict_xx finish")

        meta_000 = cat_metapath(dl, meta_00, meta_00, start_node_type=0)
        meta_001 = cat_metapath(dl, meta_00, meta_01, start_node_type=0)
        meta_010 = cat_metapath(dl, meta_01, meta_10, start_node_type=0)
        meta_011 = cat_metapath(dl, meta_01, meta_11, start_node_type=0)
        meta_101 = cat_metapath(dl, meta_10, meta_01, start_node_type=1)
        meta_111 = cat_metapath(dl, meta_11, meta_11, start_node_type=1)
        print("get_meta_dict_xxx finish")

        meta_0001 = cat_metapath(dl, meta_000, meta_01, start_node_type=0)
        meta_0011 = cat_metapath(dl, meta_001, meta_11, start_node_type=0)
        meta_0101 = cat_metapath(dl, meta_010, meta_01, start_node_type=0)
        meta_0111 = cat_metapath(dl, meta_011, meta_11, start_node_type=0)
        print("get_meta_dict_xxxx finish")

        adjlist00, idx00 = get_adjlist_idx(meta_00, meta_len=2)
        adjlist000, idx000 = get_adjlist_idx(meta_000, meta_len=3)
        adjlist010, idx010 = get_adjlist_idx(meta_010, meta_len=3)
        del meta_00, meta_000, meta_010
        print("get_adjlist_0x0 finish")

        adjlist11, idx11 = get_adjlist_idx(meta_11, meta_len=2)
        adjlist111, idx111 = get_adjlist_idx(meta_111, meta_len=3)
        adjlist101, idx101 = get_adjlist_idx(meta_101, meta_len=3)
        del meta_11, meta_111, meta_101
        print("get_adjlist_1x1 finish")

        _, idx0001 = get_adjlist_idx_condition(meta_0001, meta_len=4)  # !!!
        _, idx0011 = get_adjlist_idx_condition(meta_0011, meta_len=4)
        _, idx0101 = get_adjlist_idx_condition(meta_0101, meta_len=4)
        _, idx0111 = get_adjlist_idx_condition(meta_0111, meta_len=4)
        del meta_0001, meta_0011, meta_0101, meta_0111
        print("get_adjlist_0xx1 finish")

        adjlists_ua, edge_metapath_indices_list_ua = [[adjlist00, adjlist000, adjlist010], [adjlist11, adjlist111, adjlist101]], \
                                                     [[idx00, idx000, idx010], [idx11, idx111, idx101]]
        edge_metapath_indices_list_01 = [
            idx0001, idx0011, idx0101, idx0111
        ]
        with open(f"../../data/{dataset}/adj_and_idx1.pkl", 'wb') as file:
            pickle.dump((adjlists_ua, edge_metapath_indices_list_ua), file)
        with open(f"../../data/{dataset}/adj2.pkl", 'wb') as file:
            pickle.dump(edge_metapath_indices_list_01, file)

    adjM = sum(dl.links['data'].values())
    adjM = adjM + adjM.T
    adjM.data[:] = 1
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i

    return adjlists_ua, edge_metapath_indices_list_ua, adjM, type_mask, dl, edge_metapath_indices_list_01


def load_Zou_data(dataset='Zou'):
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/'+dataset, use_unknown_link=True)
    if os.path.exists(f"../../data/{dataset}/adj_and_idx1.pkl"):
    # if False:
        with open(f"../../data/{dataset}/adj_and_idx1.pkl", 'rb') as file:
            adjlists_ua, edge_metapath_indices_list_ua = pickle.load(file)
        print(f"loading data/{dataset}/adj_and_idx1.pkl")
        # adjlists_ua, edge_metapath_indices_list_ua = None, None

        with open(f"../../data/{dataset}/adj2.pkl", 'rb') as file:
            edge_metapath_indices_list_01 = pickle.load(file)
        print(f"loading data/{dataset}/adj2.pkl")
    else:
        meta_01 = get_meta_dict_01(dl, (0, 1))
        meta_02 = get_meta_dict_01(dl, (0, 2))
        meta_03 = get_meta_dict_01(dl, (0, 3))
        meta_04 = get_meta_dict_01(dl, (0, 4))
        meta_05 = get_meta_dict_01(dl, (0, 5))
        meta_07 = get_meta_dict_01(dl, (0, 7))

        meta_12 = get_meta_dict_01(dl, (1, 2))
        meta_13 = get_meta_dict_01(dl, (1, 3))
        meta_15 = get_meta_dict_01(dl, (1, 5))
        meta_16 = get_meta_dict_01(dl, (1, 6))
        meta_17 = get_meta_dict_01(dl, (1, 7))

        meta_23 = get_meta_dict_01(dl, (2, 3))
        meta_26 = get_meta_dict_01(dl, (2, 6))
        meta_35 = get_meta_dict_01(dl, (3, 5))
        meta_42 = get_meta_dict_01(dl, (4, 2))
        meta_43 = get_meta_dict_01(dl, (4, 3))
        meta_45 = get_meta_dict_01(dl, (4, 5))

        meta_10 = sym_metapath(dl, meta_01, end_node_type=1)
        meta_20 = sym_metapath(dl, meta_02, end_node_type=2)
        meta_21 = sym_metapath(dl, meta_12, end_node_type=2)
        meta_30 = sym_metapath(dl, meta_03, end_node_type=3)
        meta_31 = sym_metapath(dl, meta_13, end_node_type=3)
        meta_32 = sym_metapath(dl, meta_23, end_node_type=3)
        meta_40 = sym_metapath(dl, meta_04, end_node_type=4)
        meta_50 = sym_metapath(dl, meta_05, end_node_type=5)
        meta_51 = sym_metapath(dl, meta_15, end_node_type=5)
        meta_53 = sym_metapath(dl, meta_35, end_node_type=5)
        meta_61 = sym_metapath(dl, meta_16, end_node_type=6)
        meta_70 = sym_metapath(dl, meta_07, end_node_type=7)
        meta_71 = sym_metapath(dl, meta_17, end_node_type=7)

        meta_010 = cat_metapath(dl, meta_01, meta_10, start_node_type=0)
        meta_020 = cat_metapath(dl, meta_02, meta_20, start_node_type=0)
        meta_030 = cat_metapath(dl, meta_03, meta_30, start_node_type=0)
        meta_040 = cat_metapath(dl, meta_04, meta_40, start_node_type=0)
        meta_050 = cat_metapath(dl, meta_05, meta_50, start_node_type=0)
        meta_070 = cat_metapath(dl, meta_07, meta_70, start_node_type=0)
        del meta_20, meta_30, meta_40, meta_50, meta_70

        meta_101 = cat_metapath(dl, meta_10, meta_01, start_node_type=1)
        meta_121 = cat_metapath(dl, meta_12, meta_21, start_node_type=1)
        meta_131 = cat_metapath(dl, meta_13, meta_31, start_node_type=1)
        meta_151 = cat_metapath(dl, meta_15, meta_51, start_node_type=1)
        meta_161 = cat_metapath(dl, meta_16, meta_61, start_node_type=1)
        meta_171 = cat_metapath(dl, meta_17, meta_71, start_node_type=1)
        del meta_10, meta_12, meta_13, meta_15, meta_16, meta_17

        meta_231 = cat_metapath(dl, meta_23, meta_31, start_node_type=2)
        meta_261 = cat_metapath(dl, meta_26, meta_61, start_node_type=2)
        meta_321 = cat_metapath(dl, meta_32, meta_21, start_node_type=3)
        meta_351 = cat_metapath(dl, meta_35, meta_51, start_node_type=3)
        meta_421 = cat_metapath(dl, meta_42, meta_21, start_node_type=4)
        meta_431 = cat_metapath(dl, meta_43, meta_31, start_node_type=4)
        meta_451 = cat_metapath(dl, meta_45, meta_51, start_node_type=4)
        meta_531 = cat_metapath(dl, meta_53, meta_31, start_node_type=5)
        del meta_23, meta_26, meta_32, meta_35, meta_42, meta_43, meta_45, meta_53, meta_61

        meta_021 = cat_metapath(dl, meta_02, meta_21, start_node_type=0)
        meta_031 = cat_metapath(dl, meta_03, meta_31, start_node_type=0)
        meta_051 = cat_metapath(dl, meta_05, meta_51, start_node_type=0)
        meta_071 = cat_metapath(dl, meta_07, meta_71, start_node_type=0)
        del meta_21, meta_31, meta_51, meta_71

        meta_0101 = cat_metapath(dl, meta_01, meta_101, start_node_type=0)
        meta_0231 = cat_metapath(dl, meta_02, meta_231, start_node_type=0)
        meta_0261 = cat_metapath(dl, meta_02, meta_261, start_node_type=0)
        meta_0321 = cat_metapath(dl, meta_03, meta_321, start_node_type=0)
        meta_0351 = cat_metapath(dl, meta_03, meta_351, start_node_type=0)
        del meta_01, meta_02, meta_231, meta_261, meta_03, meta_321, meta_351
        meta_0421 = cat_metapath(dl, meta_04, meta_421, start_node_type=0)
        meta_0431 = cat_metapath(dl, meta_04, meta_431, start_node_type=0)
        meta_0451 = cat_metapath(dl, meta_04, meta_451, start_node_type=0)
        meta_0531 = cat_metapath(dl, meta_05, meta_531, start_node_type=0)
        del meta_04, meta_421, meta_431, meta_451, meta_05, meta_531

        adjlist010, idx010 = get_adjlist_idx(meta_010, meta_len=3)
        adjlist020, idx020 = get_adjlist_idx(meta_020, meta_len=3)
        adjlist030, idx030 = get_adjlist_idx(meta_030, meta_len=3)
        adjlist040, idx040 = get_adjlist_idx(meta_040, meta_len=3)
        adjlist050, idx050 = get_adjlist_idx(meta_050, meta_len=3)
        adjlist070, idx070 = get_adjlist_idx(meta_070, meta_len=3)
        del meta_010, meta_020, meta_030, meta_040, meta_050, meta_070
        print("get_adjlist_0x0 finish")

        adjlist101, idx101 = get_adjlist_idx(meta_101, meta_len=3)
        adjlist121, idx121 = get_adjlist_idx(meta_121, meta_len=3)
        adjlist131, idx131 = get_adjlist_idx(meta_131, meta_len=3)
        adjlist151, idx151 = get_adjlist_idx(meta_151, meta_len=3)
        adjlist161, idx161 = get_adjlist_idx(meta_161, meta_len=3)
        adjlist171, idx171 = get_adjlist_idx(meta_171, meta_len=3)
        del meta_101, meta_121, meta_131, meta_151, meta_161, meta_171
        print("get_adjlist_1x1 finish")

        _, idx021 = get_adjlist_idx(meta_021, meta_len=3)
        _, idx031 = get_adjlist_idx(meta_031, meta_len=3)
        _, idx051 = get_adjlist_idx(meta_051, meta_len=3)
        _, idx071 = get_adjlist_idx(meta_071, meta_len=3)
        del meta_021, meta_031, meta_051, meta_071
        print("get_adjlist_0x1 finish")

        _, idx0101 = get_adjlist_idx_condition(meta_0101, meta_len=4)  # !!!
        _, idx0231 = get_adjlist_idx(meta_0231, meta_len=4)
        _, idx0261 = get_adjlist_idx(meta_0261, meta_len=4)
        _, idx0321 = get_adjlist_idx(meta_0321, meta_len=4)
        del meta_0101, meta_0231, meta_0261, meta_0321

        _, idx0351 = get_adjlist_idx(meta_0351, meta_len=4)
        _, idx0421 = get_adjlist_idx(meta_0421, meta_len=4)
        _, idx0431 = get_adjlist_idx(meta_0431, meta_len=4)
        _, idx0451 = get_adjlist_idx(meta_0451, meta_len=4)
        _, idx0531 = get_adjlist_idx(meta_0531, meta_len=4)
        del meta_0351, meta_0421, meta_0431, meta_0451, meta_0531
        print("get_adjlist_0xx1 finish")

        adjlists_ua = [[adjlist010, adjlist020, adjlist030, adjlist040, adjlist050, adjlist070],
                       [adjlist101, adjlist121, adjlist131, adjlist151, adjlist161, adjlist171]]
        edge_metapath_indices_list_ua = [[idx010, idx020, idx030, idx040, idx050, idx070],
                                         [idx101, idx121, idx131, idx151, idx161, idx171]]
        edge_metapath_indices_list_01 = [
            idx021, idx031, idx051, idx071,
            idx0101, idx0231, idx0261, idx0321, idx0351, idx0421, idx0431, idx0451, idx0531
        ]
        with open(f"../../data/{dataset}/adj_and_idx1.pkl", 'wb') as file:
            pickle.dump((adjlists_ua, edge_metapath_indices_list_ua), file)
        with open(f"../../data/{dataset}/adj2.pkl", 'wb') as file:
            pickle.dump(edge_metapath_indices_list_01, file)

    adjM = sum(dl.links['data'].values())
    adjM = adjM + adjM.T
    adjM.data[:] = 1
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i

    return adjlists_ua, edge_metapath_indices_list_ua, adjM, type_mask, dl, edge_metapath_indices_list_01


def load_Zou_data_old(dataset='Zou'):
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/'+dataset, use_unknown_link=True)
    if os.path.exists(f"../../data/{dataset}/adj_and_idx1.pkl"):
    # if False:
        with open(f"../../data/{dataset}/adj_and_idx1.pkl", 'rb') as file:
            adjlists_ua, edge_metapath_indices_list_ua = pickle.load(file)
        print(f"loading data/{dataset}/adj_and_idx1.pkl")
        # adjlists_ua, edge_metapath_indices_list_ua = None, None

        with open(f"../../data/{dataset}/adj2.pkl", 'rb') as file:
            edge_metapath_indices_list_01 = pickle.load(file)
        print(f"loading data/{dataset}/adj2.pkl")
    else:
        meta_01 = get_meta_dict_01(dl, (0, 1))
        meta_10 = sym_metapath(dl, meta_01, end_node_type=1)
        meta_010 = cat_metapath(dl, meta_01, meta_10, start_node_type=0)
        meta_101 = cat_metapath(dl, meta_10, meta_01, start_node_type=1)
        meta_0101 = cat_metapath(dl, meta_01, meta_101, start_node_type=0)
        del meta_01, meta_10
        print("01, 10")

        meta_02 = get_meta_dict_01(dl, (0, 2))
        meta_20 = sym_metapath(dl, meta_02, end_node_type=2)
        meta_020 = cat_metapath(dl, meta_02, meta_20, start_node_type=0)
        del meta_20
        print("20")

        meta_12 = get_meta_dict_01(dl, (1, 2))
        meta_21 = sym_metapath(dl, meta_12, end_node_type=2)
        meta_121 = cat_metapath(dl, meta_12, meta_21, start_node_type=1)
        del meta_12
        print("12")

        meta_021 = cat_metapath(dl, meta_02, meta_21, start_node_type=0)
        meta_23 = get_meta_dict_01(dl, (2, 3))
        meta_32 = sym_metapath(dl, meta_23, end_node_type=3)
        meta_321 = cat_metapath(dl, meta_32, meta_21, start_node_type=3)
        del meta_32
        print("32")

        meta_13 = get_meta_dict_01(dl, (1, 3))
        meta_31 = sym_metapath(dl, meta_13, end_node_type=3)
        meta_131 = cat_metapath(dl, meta_13, meta_31, start_node_type=1)
        del meta_13
        meta_231 = cat_metapath(dl, meta_23, meta_31, start_node_type=2)
        del meta_23
        print("23")

        meta_03 = get_meta_dict_01(dl, (0, 3))
        meta_031 = cat_metapath(dl, meta_03, meta_31, start_node_type=0)
        meta_0321 = cat_metapath(dl, meta_03, meta_321, start_node_type=0)
        del meta_321

        meta_30 = sym_metapath(dl, meta_03, end_node_type=3)
        meta_030 = cat_metapath(dl, meta_03, meta_30, start_node_type=0)
        del meta_30
        print("30")

        meta_43 = get_meta_dict_01(dl, (4, 3))
        meta_431 = cat_metapath(dl, meta_43, meta_31, start_node_type=4)
        del meta_43

        meta_15 = get_meta_dict_01(dl, (1, 5))
        meta_51 = sym_metapath(dl, meta_15, end_node_type=5)
        meta_151 = cat_metapath(dl, meta_15, meta_51, start_node_type=1)
        del meta_15
        print("15")

        meta_05 = get_meta_dict_01(dl, (0, 5))
        meta_50 = sym_metapath(dl, meta_05, end_node_type=5)
        meta_050 = cat_metapath(dl, meta_05, meta_50, start_node_type=0)
        del meta_50

        meta_35 = get_meta_dict_01(dl, (3, 5))
        meta_53 = sym_metapath(dl, meta_35, end_node_type=5)
        meta_351 = cat_metapath(dl, meta_35, meta_51, start_node_type=3)
        del meta_35
        print("35")

        meta_531 = cat_metapath(dl, meta_53, meta_31, start_node_type=5)
        del meta_31, meta_53

        meta_051 = cat_metapath(dl, meta_05, meta_51, start_node_type=0)
        meta_0531 = cat_metapath(dl, meta_05, meta_531, start_node_type=0)
        del meta_05, meta_531

        meta_45 = get_meta_dict_01(dl, (4, 5))
        meta_451 = cat_metapath(dl, meta_45, meta_51, start_node_type=4)
        del meta_45, meta_51
        print("45")

        meta_0351 = cat_metapath(dl, meta_03, meta_351, start_node_type=0)
        del meta_03, meta_351

        meta_42 = get_meta_dict_01(dl, (4, 2))
        meta_421 = cat_metapath(dl, meta_42, meta_21, start_node_type=4)
        del meta_21, meta_42

        meta_0231 = cat_metapath(dl, meta_02, meta_231, start_node_type=0)
        del meta_231

        meta_16 = get_meta_dict_01(dl, (1, 6))
        meta_61 = sym_metapath(dl, meta_16, end_node_type=6)
        meta_161 = cat_metapath(dl, meta_16, meta_61, start_node_type=1)
        del meta_16
        print("16")

        meta_26 = get_meta_dict_01(dl, (2, 6))
        meta_261 = cat_metapath(dl, meta_26, meta_61, start_node_type=2)
        del meta_26, meta_61

        meta_0261 = cat_metapath(dl, meta_02, meta_261, start_node_type=0)
        del meta_02, meta_261

        meta_04 = get_meta_dict_01(dl, (0, 4))
        meta_40 = sym_metapath(dl, meta_04, end_node_type=4)
        meta_040 = cat_metapath(dl, meta_04, meta_40, start_node_type=0)
        del meta_40
        print("40")

        meta_0421 = cat_metapath(dl, meta_04, meta_421, start_node_type=0)
        del meta_421

        meta_0431 = cat_metapath(dl, meta_04, meta_431, start_node_type=0)
        del meta_431

        meta_0451 = cat_metapath(dl, meta_04, meta_451, start_node_type=0)
        del meta_04, meta_451

        meta_07 = get_meta_dict_01(dl, (0, 7))
        meta_70 = sym_metapath(dl, meta_07, end_node_type=7)
        meta_070 = cat_metapath(dl, meta_07, meta_70, start_node_type=0)
        del meta_70
        print("70")

        meta_17 = get_meta_dict_01(dl, (1, 7))
        meta_71 = sym_metapath(dl, meta_17, end_node_type=7)
        meta_171 = cat_metapath(dl, meta_17, meta_71, start_node_type=1)
        del meta_17

        meta_071 = cat_metapath(dl, meta_07, meta_71, start_node_type=0)
        del meta_07
        print("07")

        adjlist010, idx010 = get_adjlist_idx(meta_010, meta_len=3)
        adjlist020, idx020 = get_adjlist_idx(meta_020, meta_len=3)
        adjlist030, idx030 = get_adjlist_idx(meta_030, meta_len=3)
        adjlist040, idx040 = get_adjlist_idx(meta_040, meta_len=3)
        adjlist050, idx050 = get_adjlist_idx(meta_050, meta_len=3)
        adjlist070, idx070 = get_adjlist_idx(meta_070, meta_len=3)
        del meta_010, meta_020, meta_030, meta_040, meta_050, meta_070
        print("get_adjlist_0x0 finish")

        adjlist101, idx101 = get_adjlist_idx(meta_101, meta_len=3)
        adjlist121, idx121 = get_adjlist_idx(meta_121, meta_len=3)
        adjlist131, idx131 = get_adjlist_idx(meta_131, meta_len=3)
        adjlist151, idx151 = get_adjlist_idx(meta_151, meta_len=3)
        adjlist161, idx161 = get_adjlist_idx(meta_161, meta_len=3)
        adjlist171, idx171 = get_adjlist_idx(meta_171, meta_len=3)
        del meta_101, meta_121, meta_131, meta_151, meta_161, meta_171
        print("get_adjlist_1x1 finish")

        _, idx021 = get_adjlist_idx(meta_021, meta_len=3)
        _, idx031 = get_adjlist_idx(meta_031, meta_len=3)
        _, idx051 = get_adjlist_idx(meta_051, meta_len=3)
        _, idx071 = get_adjlist_idx(meta_071, meta_len=3)
        del meta_021, meta_031, meta_051, meta_071
        print("get_adjlist_0x1 finish")

        _, idx0101 = get_adjlist_idx_condition(meta_0101, meta_len=4)  # !!!
        _, idx0231 = get_adjlist_idx(meta_0231, meta_len=4)
        _, idx0261 = get_adjlist_idx(meta_0261, meta_len=4)
        _, idx0321 = get_adjlist_idx(meta_0321, meta_len=4)
        del meta_0101, meta_0231, meta_0261, meta_0321

        _, idx0351 = get_adjlist_idx(meta_0351, meta_len=4)
        _, idx0421 = get_adjlist_idx(meta_0421, meta_len=4)
        _, idx0431 = get_adjlist_idx(meta_0431, meta_len=4)
        _, idx0451 = get_adjlist_idx(meta_0451, meta_len=4)
        _, idx0531 = get_adjlist_idx(meta_0531, meta_len=4)
        del meta_0351, meta_0421, meta_0431, meta_0451, meta_0531
        print("get_adjlist_0xx1 finish")

        adjlists_ua = [[adjlist010, adjlist020, adjlist030, adjlist040, adjlist050, adjlist070],
                       [adjlist101, adjlist121, adjlist131, adjlist151, adjlist161, adjlist171]]
        edge_metapath_indices_list_ua = [[idx010, idx020, idx030, idx040, idx050, idx070],
                                         [idx101, idx121, idx131, idx151, idx161, idx171]]
        edge_metapath_indices_list_01 = [
            idx021, idx031, idx051, idx071,
            idx0101, idx0231, idx0261, idx0321, idx0351, idx0421, idx0431, idx0451, idx0531
        ]
        with open(f"../../data/{dataset}/adj_and_idx1.pkl", 'wb') as file:
            pickle.dump((adjlists_ua, edge_metapath_indices_list_ua), file)
        with open(f"../../data/{dataset}/adj2.pkl", 'wb') as file:
            pickle.dump(edge_metapath_indices_list_01, file)

    adjM = sum(dl.links['data'].values())
    adjM = adjM + adjM.T
    adjM.data[:] = 1
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i

    return adjlists_ua, edge_metapath_indices_list_ua, adjM, type_mask, dl, edge_metapath_indices_list_01


def get_total_size(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, np.ndarray):
        size += obj.nbytes
        if obj.dtype == object:
            for item in obj:
                size += get_total_size(item, seen)
        return size
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += get_total_size(item, seen)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            size += get_total_size(key, seen)
            size += get_total_size(value, seen)
    return size



# def get_adjlist_pkl(dl, meta, return_dic=True, symmetric=False):
#     meta010, meta01, meta10 = dl.get_full_meta_path(meta, symmetric=symmetric)  # checked
#     adjlist00, idx00 = {}, {}
#     for k in meta010:
#         idx00[k] = np.array(sorted([tuple(reversed(i)) for i in meta010[k]]), dtype=np.int32).reshape([-1, len(meta)+1])
#         adjlist00[k] = idx00[k][:, 0].tolist()
#     if not return_dic:
#         idx00 = np.concatenate(list(idx00.values()), axis=0)
#     return adjlist00, idx00, meta01


# def get_adjlist_pkl_special(dl, meta, meta01, meta12, symmetric: bool=True, sample_rate=0.45):
#     from tqdm import tqdm
#     from collections import defaultdict
#     import random
#     if not dl.nonzero:
#         dl.get_nonzero()
#     meta = [dl.get_edge_type(x) for x in meta]
#
#     # 采样拼接：由meta01和meta12采样拼接得meta012
#     meta_012 = defaultdict(list)
#     start_node_type = dl.links['meta'][meta[0]][0] if meta[0] >= 0 else dl.links['meta'][-meta[0] - 1][1]
#     for i in tqdm(range(dl.nodes['shift'][start_node_type],
#                    dl.nodes['shift'][start_node_type] + dl.nodes['count'][start_node_type])):
#         sample_01_edges = random.sample(meta01[i], int(len(meta01[i]) * sample_rate))
#         for beg in sample_01_edges:
#             sample_12_edges = random.sample(meta12[beg[-1]], int(len(meta12[beg[-1]]) * sample_rate))
#             for end in sample_12_edges:
#                 meta_012[i].append(beg + end[1:])
#
#     # 对称：由meta012对称得到meta210
#     meta_210 = defaultdict(list)
#     if symmetric:
#         for paths in tqdm(meta_012.values()):
#             for path in paths:
#                 meta_210[path[-1]].append(path[::-1])
#     else:
#         raise   # not finish
#
#     # 拼接：拼接meta012和meta210得到meta01210
#     # mat010 = np.zeros((dl.nodes['count'][start_node_type], dl.nodes['count'][start_node_type]))
#     shift = dl.nodes['shift'][start_node_type]
#     meta_01210 = {}
#     for i in tqdm(range(shift, shift + dl.nodes['count'][start_node_type])):
#         meta_01210[i] = []
#         for beg in meta_012[i]:
#             for end in meta_210[beg[-1]]:
#                 meta_01210[i].append(beg + end[1:])
#                 # mat010[beg[0] - shift, end[-1] - shift] += 1
#     # mat010 = sp.coo_matrix(mat010)
#
#     # tt = time.time()
#     # adjlist00 = [[] for _ in range(dl.nodes['count'][0])]
#     # for i,j,v in zip(mat010.row, mat010.col, mat010.data):
#     #     adjlist00[i].extend([j]*int(v))
#     # adjlist00 = [' '.join(map(str, [i]+sorted(x))) for i,x in enumerate(adjlist00)]
#     # print(f"run special adjlist00 time = {time.time()-tt:.2f}s")
#
#     tt = time.time()
#     adjlist00, idx00 = {}, {}
#     for k in meta_01210:
#         idx00[k-dl.nodes['shift'][0]] = np.array(sorted([tuple(reversed(i)) for i in meta_01210[k]]), dtype=np.int32).reshape([-1, len(meta)+1])
#         adjlist00[k] = idx00[k][:, 0].tolist()
#     print(f"run special idx00 time = {time.time()-tt:.2f}s")
#
#     return adjlist00, idx00
#
# def get_edges_from_adjlist(adjlist: dict[list]) -> (list, list):
#     src, dst = [], []
#     for k, v in adjlist.items():
#         src.extend([k, ] * len(v))
#         dst.extend(v)
#     return src, dst
#
#
# # 适用于首尾相同节点类型的元路径生成dgl graph，且元路径长度小于5(否则图过于庞大)
# def get_fixed_dgl_graph(adjlists_ua: list[list], edge_metapath_indices_list_ua: list[list], expected_metapaths: list[list], device):
#     dgl_graph_list, metapath_indices_list = [], []
#     for i, (node_adjlist, node_metapaths, node_id_path) in enumerate(zip(adjlists_ua, edge_metapath_indices_list_ua, expected_metapaths)):
#         dgl_graph_list.append([])
#         metapath_indices_list.append([])
#         for j, (adjlist, metapath, id_path) in enumerate(zip(node_adjlist, node_metapaths, node_id_path)):
#             if len(id_path) >= 5:
#                 dgl_graph_list[i].append(None)
#                 metapath_indices_list[i].append(None)
#             else:
#                 src, dst = get_edges_from_adjlist(adjlist)
#                 g = dgl.graph((src, dst)).to(device)
#                 dgl_graph_list[i].append(g)
#                 metapath_indice = np.vstack(list(metapath.values()))
#                 metapath_indices_list[i].append(torch.LongTensor(metapath_indice).to(device))
#     return dgl_graph_list, metapath_indices_list
#
#
# def get_random_dgl_graph(adjlist: dict[list], metapath_indice: dict[np.ndarray], max_neigh_num=500):
#     adjlist_sample, metapath_indice_sample = sample_dicts(adjlist, metapath_indice, max_neigh_num)
#     src, dst = get_edges_from_adjlist(adjlist_sample)
#     result_metapath_indice = np.vstack(list(metapath_indice_sample.values()))
#     result_metapath_indice = torch.LongTensor(result_metapath_indice)
#     assert len(src) == len(dst)
#     assert len(src) == result_metapath_indice.shape[0]
#     g = dgl.graph((src, dst))
#
#     return g, result_metapath_indice
#
#
# def sample_dicts(dict_A: dict, dict_B: dict, n) -> (dict, dict):
#     sampled_A, sampled_B = {}, {}
#
#     for key in dict_A:
#         array_A = np.array(dict_A[key])
#         array_B = dict_B[key]
#
#         m = array_A.shape[0]
#         if array_B.shape[0] != m:
#             raise ValueError(f"对于 key '{key}'，dict_A 和 dict_B 的长度不匹配。")
#
#         if m <= n:
#             sampled_A[key] = array_A.tolist()
#             sampled_B[key] = array_B
#         else:
#             indices = np.random.choice(m, size=n, replace=False)
#             sampled_A[key] = array_A[indices].tolist()
#             sampled_B[key] = array_B[indices]
#
#     return sampled_A, sampled_B
#
#
# def complete_none_list(dgl_graph_list: list[list], metapath_indices_list: list[list], adjlists_ua, edge_metapath_indices_list_ua, device, max_neigh_num=500):
#     new_g_list = copy.deepcopy(dgl_graph_list)
#     new_m_list = copy.deepcopy(metapath_indices_list)
#     for i, (node_graph, node_metapath) in enumerate(zip(dgl_graph_list, metapath_indices_list)):
#         for j, (graph, metapath) in enumerate(zip(node_graph, node_metapath)):
#             if graph is None:
#                 assert metapath is None
#                 g, metapath_indice = get_random_dgl_graph(adjlists_ua[i][j], edge_metapath_indices_list_ua[i][j], max_neigh_num)
#                 new_g_list[i][j] = g.to(device)
#                 new_m_list[i][j] = metapath_indice.to(device)
#     return new_g_list, new_m_list


'''
def load_Freebase_data():
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/Freebase')
    import json
    adjlists, idxs = [], []
    for fn in ['meta1.json', 'meta2.json']:
      with open('meta1.json', 'r', encoding='utf-8') as f:
        meta = json.loads(''.join(f.readlines()))
      for i, x in enumerate(meta['node_0'][:5]):
        path = list(map(int, x['path'].split(',')))
        path = path + [-x-1 for x in reversed(path)]
        th_adj, th_idx = get_adjlist_pkl(dl, path)
        adjlists.append(th_adj)
        idxs.append(th_idx)
        print('meta path {}-{} done'.format(fn, i))
    features = []
    types = len(dl.nodes['count'])
    for i in range(types):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
        else:
            if type(th) is np.ndarray:
                features.append(th)
            else:
                features.append(th.toarray())
    #features_0, features_1, features_2, features_3 = features
    adjM = sum(dl.links['data'].values())
    adjM = adjM + adjM.T
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(types):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return adjlists, \
           idxs, \
           features, \
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx,\
            dl


def load_ACM_data():
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/ACM')
    dl.get_sub_graph([0,1,2])
    #dl.links['data'][0] += sp.eye(dl.nodes['total'])
    for i in range(dl.nodes['count'][0]):
        if dl.links['data'][0][i].sum() == 0:
            dl.links['data'][0][i,i] = 1
        if dl.links['data'][1][i].sum() == 0:
            dl.links['data'][1][i,i] = 1
    adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)])
    print('meta path 1 done')
    adjlist01, idx01 = get_adjlist_pkl(dl, [(0,2), (2,0)])
    print('meta path 2 done')
    adjlist02, idx02 = get_adjlist_pkl(dl, [0, (0,1), (1,0)])
    print('meta path 3 done')
    adjlist03, idx03 = get_adjlist_pkl(dl, [0, (0,2), (2,0)])
    print('meta path 4 done')
    adjlist04, idx04 = get_adjlist_pkl(dl, [1, (0,1), (1,0)])
    print('meta path 5 done')
    adjlist05, idx05 = get_adjlist_pkl(dl, [1, (0,2), (2,0)])
    print('meta path 6 done')
    features = []
    types = len(dl.nodes['count'])
    for i in range(types):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
        else:
            if type(th) is np.ndarray:
                features.append(th)
            else:
                features.append(th.toarray())
    #features_0, features_1, features_2, features_3 = features
    adjM = sum(dl.links['data'].values())
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(types):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return [adjlist00, adjlist01, adjlist02, adjlist03, adjlist04, adjlist05], \
           [idx00, idx01, idx02, idx03, idx04, idx05], \
           features, \
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx,\
            dl


def load_DBLP_data():
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/DBLP')
    adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)])
    print('meta path 1 done')
    adjlist01, idx01 = get_adjlist_pkl(dl, [(0,1), (1,2), (2,1), (1,0)])
    print('meta path 2 done')
    adjlist02, idx02 = get_adjlist_pkl(dl, [(0,1), (1,3), (3,1), (1,0)])
    print('meta path 3 done')
    features = []
    for i in range(4):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
        else:
            if type(th) is np.ndarray:
                features.append(th)
            else:
                features.append(th.toarray())
    features_0, features_1, features_2, features_3 = features
    adjM = sum(dl.links['data'].values())
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(4):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    """
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    features_3 = np.eye(20, dtype=np.float32)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    """
    return [adjlist00, adjlist01, adjlist02], \
           [idx00, idx01, idx02], \
           [features_0, features_1, features_2, features_3],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx,\
            dl


def load_IMDB_data(prefix='data/preprocessed/IMDB_processed'):
    G00 = nx.read_adjlist(prefix + '/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
    G01 = nx.read_adjlist(prefix + '/0/0-2-0.adjlist', create_using=nx.MultiDiGraph)
    G10 = nx.read_adjlist(prefix + '/1/1-0-1.adjlist', create_using=nx.MultiDiGraph)
    G11 = nx.read_adjlist(prefix + '/1/1-0-2-0-1.adjlist', create_using=nx.MultiDiGraph)
    G20 = nx.read_adjlist(prefix + '/2/2-0-2.adjlist', create_using=nx.MultiDiGraph)
    G21 = nx.read_adjlist(prefix + '/2/2-0-1-0-2.adjlist', create_using=nx.MultiDiGraph)
    idx00 = np.load(prefix + '/0/0-1-0_idx.npy')
    idx01 = np.load(prefix + '/0/0-2-0_idx.npy')
    idx10 = np.load(prefix + '/1/1-0-1_idx.npy')
    idx11 = np.load(prefix + '/1/1-0-2-0-1_idx.npy')
    idx20 = np.load(prefix + '/2/2-0-2_idx.npy')
    idx21 = np.load(prefix + '/2/2-0-1-0-2_idx.npy')
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz')
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz')
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz')
    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    return [[G00, G01], [G10, G11], [G20, G21]], \
           [[idx00, idx01], [idx10, idx11], [idx20, idx21]], \
           [features_0, features_1, features_2],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx


def load_IMDB_data_new():
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/IMDB')
    adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)], 0, False)
    G00 = nx.readwrite.adjlist.parse_adjlist(adjlist00, create_using=nx.MultiDiGraph)
    print('meta path 1 done')
    adjlist01, idx01 = get_adjlist_pkl(dl, [(0,2), (2,0)], 0, False)
    G01 = nx.readwrite.adjlist.parse_adjlist(adjlist01, create_using=nx.MultiDiGraph)
    print('meta path 2 done')
    adjlist10, idx10 = get_adjlist_pkl(dl, [(1,0), (0,1)], 1, False)
    G10 = nx.readwrite.adjlist.parse_adjlist(adjlist10, create_using=nx.MultiDiGraph)
    print('meta path 3 done')
    adjlist11, idx11 = get_adjlist_pkl(dl, [(1,0), (0,2), (2,0), (0, 1)], 1, False)
    G11 = nx.readwrite.adjlist.parse_adjlist(adjlist11, create_using=nx.MultiDiGraph)
    print('meta path 4 done')
    adjlist20, idx20 = get_adjlist_pkl(dl, [(2,0), (0,2)], 2, False)
    G20 = nx.readwrite.adjlist.parse_adjlist(adjlist20, create_using=nx.MultiDiGraph)
    print('meta path 5 done')
    adjlist21, idx21 = get_adjlist_pkl(dl, [(2,0), (0,1), (1,0), (0,2)], 2, False)
    G21 = nx.readwrite.adjlist.parse_adjlist(adjlist21, create_using=nx.MultiDiGraph)
    print('meta path 6 done')
    features = []
    types = len(dl.nodes['count'])
    for i in range(types):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
        else:
            if type(th) is np.ndarray:
                features.append(th)
            else:
                features.append(th.toarray())
    #features_0, features_1, features_2, features_3 = features
    adjM = sum(dl.links['data'].values())
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(types):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    #labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return [[G00, G01], [G10, G11], [G20, G21]], \
           [[idx00, idx01], [idx10, idx11], [idx20, idx21]], \
           features, \
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx,\
            dl


# load skipgram-format embeddings, treat missing node embeddings as zero vectors
def load_skipgram_embedding(path, num_embeddings):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings = np.zeros((num_embeddings, dim))
        for line in infile.readlines():
            count += 1
            line = line.strip().split(' ')
            embeddings[int(line[0])] = np.array(list(map(float, line[1:])))
    print('{} out of {} nodes have non-zero embeddings'.format(count, num_embeddings))
    return embeddings


# load metapath2vec embeddings
def load_metapath2vec_embedding(path, type_list, num_embeddings_list, offset_list):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings_dict = {type: np.zeros((num_embeddings, dim)) for type, num_embeddings in zip(type_list, num_embeddings_list)}
        offset_dict = {type: offset for type, offset in zip(type_list, offset_list)}
        for line in infile.readlines():
            line = line.strip().split(' ')
            # drop </s> token
            if line[0] == '</s>':
                continue
            count += 1
            embeddings_dict[line[0][0]][int(line[0][1:]) - offset_dict[line[0][0]]] = np.array(list(map(float, line[1:])))
    print('{} node embeddings loaded'.format(count))
    return embeddings_dict


def load_glove_vectors(dim=50):
    print('Loading GloVe pretrained word vectors')
    file_paths = {
        50: 'data/wordvec/GloVe/glove.6B.50d.txt',
        100: 'data/wordvec/GloVe/glove.6B.100d.txt',
        200: 'data/wordvec/GloVe/glove.6B.200d.txt',
        300: 'data/wordvec/GloVe/glove.6B.300d.txt'
    }
    f = open(file_paths[dim], 'r', encoding='utf-8')
    wordvecs = {}
    for line in f.readlines():
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        wordvecs[word] = embedding
    print('Done.', len(wordvecs), 'words loaded!')
    return wordvecs
'''
