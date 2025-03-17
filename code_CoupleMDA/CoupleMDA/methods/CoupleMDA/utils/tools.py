import pandas as pd
import torch
import dgl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
import scipy.sparse as sp
from copy import deepcopy
from scipy.sparse import csr_matrix
from collections import deque


def parse_adjlist_LastFM(keys, adjlist, edge_metapath_indices, samples=None, exclude=None, mode=None):
    edges = np.empty(shape=(0, 2))
    if exclude is not None:
        exclude = set(map(tuple, exclude))
    result_indices = []
    for key, row, indices in zip(keys, adjlist, edge_metapath_indices):
        none_neigh = False
        row_parsed = np.array(row)
        if row_parsed.shape[0] == 0:
            none_neigh = True
        else:
            if samples is None:
                if exclude is not None:
                    if mode == 0:
                        mask = [False if (u1, a1) in exclude or (u2, a2) in exclude else True for u1, a1, u2, a2 in indices[:, [0, 1, -1, -2]]]
                    else:
                        mask = [False if (u1, a1) in exclude or (u2, a2) in exclude else True for a1, u1, a2, u2 in indices[:, [0, 1, -1, -2]]]
                    if not any(mask):  # all False
                        none_neigh = True
                    else:
                        neighbors = row_parsed[mask]
                        result_indices.append(indices[mask])
                else:
                    neighbors = row_parsed
                    result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                # _, counts = np.unique(row_parsed[1:], return_counts=True)
                # p = []
                # for count in counts:
                #     p += [(count ** (3 / 4)) / count] * count
                # p = np.array(p)
                # p = p / p.sum()
                sample = min(samples, row_parsed.shape[0])
                # sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))   # 按概率采样
                sampled_idx = np.random.choice(row_parsed.shape[0], sample, replace=False)
                if exclude is not None:
                    if mode == 0:
                        mask = [False if (u1, a1) in exclude or (u2, a2) in exclude else True for u1, a1, u2, a2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    else:
                        mask = [False if (u1, a1) in exclude or (u2, a2) in exclude else True for a1, u1, a2, u2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    if not any(mask):   # all False
                        none_neigh = True
                    else:
                        neighbors = row_parsed[sampled_idx][mask]
                        result_indices.append(indices[sampled_idx][mask])
                else:
                    neighbors = row_parsed[sampled_idx]
                    result_indices.append(indices[sampled_idx])
        if none_neigh:
            neighbors = np.array([key])
            indices = np.array([[key] * indices.shape[1]])
            result_indices.append(indices)

        edges = np.vstack((edges, np.column_stack((neighbors, [key]*neighbors.shape[0]))))
    #     for dst in neighbors:
    #         nodes.add(dst)
    #         edges.append((key, dst))
    # mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    # edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges.astype(int), result_indices


def parse_minibatch_LastFM(adjlists_ua, edge_metapath_indices_list_ua,
                           adjlists_01, edge_metapath_indices_list_01, adjM: sp.csr_matrix,
                           user_artist_batch: np.ndarray, device, samples=None, use_masks=None, test_mode=False):
    g_lists = [[], []]
    result_indices_lists = [[], []]
    target_idx_list = []
    batch_nodes = [list(set(user_artist_batch[:, 0])), list(set(user_artist_batch[:, 1]))]
    for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(adjlists_ua, edge_metapath_indices_list_ua)):
        target_idx_list.append(user_artist_batch[:, mode].astype(int))
        for adjlist, indices, use_mask in zip(adjlists, edge_metapath_indices_list, use_masks[mode]):
            if use_mask:
                edges, result_indices = parse_adjlist_LastFM(
                    batch_nodes[mode], [adjlist[node] for node in batch_nodes[mode]], [indices[node] for node in batch_nodes[mode]], samples, user_artist_batch, mode)
            else:
                edges, result_indices = parse_adjlist_LastFM(
                    batch_nodes[mode], [adjlist[node] for node in batch_nodes[mode]], [indices[node] for node in batch_nodes[mode]], samples, mode=mode)
            g = dgl.graph((edges[:, 0], edges[:, 1]))
            result_indices = torch.LongTensor(result_indices).to(device)

            if max(user_artist_batch[:, mode]) >= g.num_nodes():
                raise IndexError

            g_lists[mode].append(g.to(device))
            result_indices_lists[mode].append(result_indices)

    shift_list = []
    link_indeices_list = []
    link_count_list = []
    for edge_metapath_indices_01 in edge_metapath_indices_list_01:
        shift, result_indices, link_count = parse_adjlist_01(
            user_artist_batch, [edge_metapath_indices_01[row[0]] for row in user_artist_batch])
        result_indices = torch.LongTensor(result_indices)

        shift_list.append(shift)
        link_indeices_list.append(result_indices.to(device))
        link_count_list.append(link_count)
    link_counts = torch.stack(link_count_list, dim=1).to(device)

    # adjm = filter_csr_matrix(adjM, user_artist_batch)
    # iden_g = csr_to_dgl_graph(adjm, user_artist_batch, test_mode=test_mode).to(device)
    iden_g = None

    return g_lists, result_indices_lists, target_idx_list, shift_list, link_indeices_list, link_counts, iden_g


def parse_minibatch_LastFM_01(edge_metapath_indices_list_01, user_artist_batch: np.ndarray, device):
    shift_list = []
    link_indeices_list = []
    link_count_list = []
    for edge_metapath_indices_01 in edge_metapath_indices_list_01:
        shift, result_indices, link_count = parse_adjlist_01(
            user_artist_batch, [edge_metapath_indices_01[row[0]] for row in user_artist_batch])
        result_indices = torch.LongTensor(result_indices)

        shift_list.append(shift)
        link_indeices_list.append(result_indices.to(device))
        link_count_list.append(link_count)
    link_counts = torch.stack(link_count_list, dim=1).to(device)

    return shift_list, link_indeices_list, link_counts


def parse_adjlist_01(user_artist_batch: np.ndarray, edge_metapath_indices: list[np.ndarray], max_path=1000):
    shift = [0]
    link_count_list = []
    result_indices = np.empty(shape=(0, edge_metapath_indices[0].shape[1]))
    for row, indices in zip(user_artist_batch, edge_metapath_indices):
        link_metapath = indices[indices[:, 0] == row[1]]
        link_count = link_metapath.shape[0]
        link_count_list.append(link_count)
        if link_count:
            if link_count > max_path:
                link_metapath = link_metapath[:max_path]
                link_count = max_path
            shift.append(shift[-1] + link_count)
            result_indices = np.vstack((result_indices, link_metapath))
        else:
            shift.append(shift[-1] + 1)
            result_indices = np.vstack((result_indices, np.array([row[1]] * 2 + [row[0]] * (indices.shape[1] - 2))))
    assert shift[-1] == result_indices.shape[0]

    return shift, result_indices.astype(int), torch.tensor(link_count_list).float()


def filter_csr_matrix(csr_matrix, node_idx: np.ndarray):
    node_set = list(set(node_idx.flatten()))
    rows, cols = csr_matrix.nonzero()
    mask = np.isin(rows, node_set) | np.isin(cols, node_set)
    filtered_rows, filtered_cols = rows[mask], cols[mask]
    filtered_data = csr_matrix.data[mask]
    new_csr_matrix = sp.csr_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=csr_matrix.shape)
    return new_csr_matrix


def csr_to_dgl_graph(csr_matrix: sp.csr_matrix, mask_idx, test_mode=False,
                     add_self_loop=True, directed_opera=False):
    mat = deepcopy(csr_matrix)
    if not test_mode and mask_idx is not None:
        mat[mask_idx[:, 0], mask_idx[:, 1]] = 0
        mat[mask_idx[:, 1], mask_idx[:, 0]] = 0
    src, dst = mat.nonzero()
    src, dst = src.tolist(), dst.tolist()
    if directed_opera:
        src += dst
        dst += src
    g = dgl.graph((src, dst), num_nodes=mat.shape[0])
    if add_self_loop:
        g = dgl.add_self_loop(g)
    return g


def sort_ndarray(arr):
    sorted_indices = np.lexsort((arr[:, 1], arr[:, 0]))
    sorted_arr = arr[sorted_indices]
    return sorted_arr


def get_node_metapath_2_edges(edge_metapath_indices_list_12, nodes: list[int], save_file=None):
    assert len(nodes) == 2

    edges = set()
    for metapath in edge_metapath_indices_list_12:
        metapath0 = metapath[nodes[0]]
        link_metapath = metapath0[metapath0[:, 0] == nodes[1]]
        for j in range(link_metapath.shape[0]):
            for k in range(1, link_metapath.shape[1]):
                edges.add((link_metapath[j, k-1], link_metapath[j, k]))
    edges = [list(tup) for tup in edges]

    if save_file is None:
        return np.array(list(edges))
    else:
        df = pd.DataFrame(list(edges), columns=["source", "target"])
        df.to_csv(save_file, index=False, header=True)


def find_nth_order_neighbors(adj_matrix, node1, node2, n, output_file):
    # 用来存储最终的边集合，避免重复
    edges = set()

    # 当前层的邻居
    current_neighbors = {node1, node2}

    for step in range(n):
        next_neighbors = set()

        # 获取当前层每个节点的邻居
        for node in current_neighbors:
            row_start = adj_matrix.indptr[node]
            row_end = adj_matrix.indptr[node + 1]
            neighbors = adj_matrix.indices[row_start:row_end]

            # 将当前节点和其邻居的边加入集合
            for neighbor in neighbors:
                if step < n - 1:  # 直到n-1阶邻居，不加入最终集合
                    next_neighbors.add(neighbor)
                # 为了避免重复边，只将边加入集合
                edges.add(tuple(sorted([node, neighbor])))  # 按照升序排列，确保无重复

        current_neighbors = next_neighbors

    # 将边存储为DataFrame格式
    edge_list = list(edges)
    df = pd.DataFrame(edge_list, columns=['source', 'target'])

    # 输出为CSV文件
    df.to_csv(output_file, index=False)
    print(f"Edges saved to 'f{output_file}'")



# class index_generator:
#     def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
#         if num_data is not None:
#             self.num_data = num_data
#             self.indices = np.arange(num_data)
#         if indices is not None:
#             self.num_data = len(indices)
#             self.indices = np.copy(indices)
#         self.batch_size = batch_size
#         self.iter_counter = 0
#         self.shuffle = shuffle
#         if shuffle:
#             np.random.shuffle(self.indices)
#
#     def next(self):
#         if self.num_iterations_left() <= 0:
#             self.reset()
#         self.iter_counter += 1
#         return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])
#
#     def num_iterations(self):
#         return int(np.ceil(self.num_data / self.batch_size))
#
#     def num_iterations_left(self):
#         return self.num_iterations() - self.iter_counter
#
#     def reset(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)
#         self.iter_counter = 0
#
#
# def idx_to_one_hot(idx_arr):
#     one_hot = np.zeros((idx_arr.shape[0], idx_arr.max() + 1))
#     one_hot[np.arange(idx_arr.shape[0]), idx_arr] = 1
#     return one_hot
#
#
# def kmeans_test(X, y, n_clusters, repeat=10):
#     nmi_list = []
#     ari_list = []
#     for _ in range(repeat):
#         kmeans = KMeans(n_clusters=n_clusters)
#         y_pred = kmeans.fit_predict(X)
#         nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
#         ari_score = adjusted_rand_score(y, y_pred)
#         nmi_list.append(nmi_score)
#         ari_list.append(ari_score)
#     return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)
#
#
# def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
#     random_states = [182318 + i for i in range(repeat)]
#     result_macro_f1_list = []
#     result_micro_f1_list = []
#     for test_size in test_sizes:
#         macro_f1_list = []
#         micro_f1_list = []
#         for i in range(repeat):
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
#             svm = LinearSVC(dual=False)
#             svm.fit(X_train, y_train)
#             y_pred = svm.predict(X_test)
#             macro_f1 = f1_score(y_test, y_pred, average='macro')
#             micro_f1 = f1_score(y_test, y_pred, average='micro')
#             macro_f1_list.append(macro_f1)
#             micro_f1_list.append(micro_f1)
#         result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
#         result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
#     return result_macro_f1_list, result_micro_f1_list
#
#
# def evaluate_results_nc(embeddings, labels, num_classes):
#     print('SVM test')
#     svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels)
#     print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(macro_f1_mean, macro_f1_std, train_size) for
#                                     (macro_f1_mean, macro_f1_std), train_size in
#                                     zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
#     print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
#                                     (micro_f1_mean, micro_f1_std), train_size in
#                                     zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
#     print('K-means test')
#     nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)
#     print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
#     print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))
#
#     return svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std
#
#
# def parse_adjlist(adjlist, edge_metapath_indices, samples=None):
#     edges = []
#     nodes = set()
#     result_indices = []
#     for row, indices in zip(adjlist, edge_metapath_indices):
#         row_parsed = list(map(int, row.split(' ')))
#         nodes.add(row_parsed[0])
#         if len(row_parsed) > 1:
#             # sampling neighbors
#             if samples is None:
#                 neighbors = row_parsed[1:]
#                 result_indices.append(indices)
#             else:
#                 # undersampling frequent neighbors
#                 unique, counts = np.unique(row_parsed[1:], return_counts=True)
#                 p = []
#                 for count in counts:
#                     p += [(count ** (3 / 4)) / count] * count
#                 p = np.array(p)
#                 p = p / p.sum()
#                 samples = min(samples, len(row_parsed) - 1)
#                 sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
#                 neighbors = [row_parsed[i + 1] for i in sampled_idx]
#                 result_indices.append(indices[sampled_idx])
#         else:
#             neighbors = []
#             result_indices.append(indices)
#         for dst in neighbors:
#             nodes.add(dst)
#             edges.append((row_parsed[0], dst))
#     mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
#     edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
#     result_indices = np.vstack(result_indices)
#     return edges, result_indices, len(nodes), mapping
#
#
# def parse_minibatch(adjlists, edge_metapath_indices_list, idx_batch, device, samples=None):
#     g_list = []
#     result_indices_list = []
#     idx_batch_mapped_list = []
#     for adjlist, indices in zip(adjlists, edge_metapath_indices_list):
#         edges, result_indices, num_nodes, mapping = parse_adjlist(
#             [adjlist[i] for i in idx_batch], [indices[i] for i in idx_batch], samples)
#
#         g = dgl.DGLGraph(multigraph=True)
#         g.add_nodes(num_nodes)
#         if len(edges) > 0:
#             sorted_index = sorted(range(len(edges)), key=lambda i : edges[i])
#             g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
#             result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
#         else:
#             result_indices = torch.LongTensor(result_indices).to(device)
#         #g.add_edges(*list(zip(*[(dst, src) for src, dst in sorted(edges)])))
#         #result_indices = torch.LongTensor(result_indices).to(device)
#         g_list.append(g)
#         result_indices_list.append(result_indices)
#         idx_batch_mapped_list.append(np.array([mapping[idx] for idx in idx_batch]))
#
#     return g_list, result_indices_list, idx_batch_mapped_list
