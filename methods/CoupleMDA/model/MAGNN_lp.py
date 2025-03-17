import time

import torch
import torch.nn as nn
import numpy as np
from methods.CoupleGNN.utils.tools import parse_minibatch_LastFM, sort_ndarray
from .base_MAGNN import MAGNN_ctr_ntype_specific
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from dgl.nn.pytorch import GATConv
from torch.utils.data import TensorDataset, DataLoader
from .dgl_hetero_GNN import HeteroGNN, HGT
import pandas as pd


class MAGNN_lp(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 etypes_01,
                 feats_dim_list,
                 hidden_dim,
                 num_heads,
                 attn_vec_dim,
                 edge_types: list,
                 f_shift: list,
                 rnn_type='gru',
                 l_enco_type='RotatE0',
                 dropout_rate=0.5,
                 num_link_type=4,
                 use_self_attn=True,
                 meta_mode='node_link',
                 inner_attn=False,
                 self_super_meta=False,
                 is_train=True,
                 use_HMPNN=True):
        super(MAGNN_lp, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_self_attn = use_self_attn
        if meta_mode not in ['node_link', 'just_node', 'just_link']:
            raise NotImplementedError
        self.meta_mode = meta_mode
        self.inner_attn = inner_attn
        self.self_super_meta = self_super_meta
        self.is_train = is_train
        self.use_HMPNN = use_HMPNN
        drfn = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()

        # etype-specific parameters
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, hidden_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, hidden_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, hidden_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, hidden_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        if self.use_HMPNN:
            self.HGCN = HeteroGNN(hidden_dim, hidden_dim, edge_types, f_shift)
        # self.HGCN = HGT(hidden_dim, hidden_dim, edge_types, f_shift)

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        self.feat_drop = nn.Identity()
        # self.feat_drop = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()
        [nn.init.xavier_normal_(fc.weight, gain=1.414) for fc in self.fc_list]

        # ctr_ntype-specific layers
        if self.meta_mode in ['node_link', 'just_node']:
            self.user_layer = MAGNN_ctr_ntype_specific(num_metapaths_list[0],
                                                       etypes_lists[0],
                                                       hidden_dim,
                                                       num_heads,
                                                       attn_vec_dim,
                                                       rnn_type,
                                                       r_vec=r_vec,
                                                       attn_drop=dropout_rate,
                                                       use_minibatch=True,
                                                       use_self_attn=self.use_self_attn,
                                                       inner_attn=self.inner_attn,
                                                       self_super_meta=self.self_super_meta)
            self.item_layer = MAGNN_ctr_ntype_specific(num_metapaths_list[1],
                                                       etypes_lists[1],
                                                       hidden_dim,
                                                       num_heads,
                                                       attn_vec_dim,
                                                       rnn_type,
                                                       r_vec=r_vec,
                                                       attn_drop=dropout_rate,
                                                       use_minibatch=True,
                                                       use_self_attn=self.use_self_attn,
                                                       inner_attn=self.inner_attn,
                                                       self_super_meta=self.self_super_meta)
            self.w_user = nn.Linear(hidden_dim*num_heads, hidden_dim)
            self.w_item = nn.Linear(hidden_dim*num_heads, hidden_dim)
            nn.init.xavier_normal_(self.w_user.weight, gain=1.414)
            nn.init.xavier_normal_(self.w_item.weight, gain=1.414)
            # self.w_xij = nn.Linear(hidden_dim*num_heads, hidden_dim)
            self.fc_xij = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU()
            )

        if self.meta_mode in ['node_link', 'just_link']:
            # r_vec_link = nn.Parameter(torch.empty(size=(num_link_type // 2, hidden_dim // 2, 2)))
            # nn.init.xavier_normal_(r_vec_link.data, gain=1.414)
            r_vec_link = None
            self.link_layer = MAGNN_ctr_ntype_specific(num_link_type,
                                                       etypes_01,
                                                       hidden_dim,
                                                       num_heads,
                                                       attn_vec_dim,
                                                       l_enco_type=l_enco_type,
                                                       r_vec=r_vec_link,
                                                       link_mode=True,
                                                       use_self_attn=self.use_self_attn,
                                                       self_super_meta=self.self_super_meta)
            self.fc_link = nn.Sequential(
                nn.Linear(hidden_dim*num_heads, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU()
            )
            # self.w_link = nn.Linear(hidden_dim*num_heads, hidden_dim)
            self.beta = nn.Parameter(torch.ones(1))
        self.norm_n = nn.LayerNorm(hidden_dim)
        self.norm_l = nn.LayerNorm(hidden_dim)

        self.lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1))

        if self.self_super_meta:
            self.fc_node_meta_type = nn.Sequential(
                nn.Linear(hidden_dim * num_heads, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU(),
                nn.Linear(hidden_dim, sum(num_metapaths_list))
            )
            self.fc_link_meta_type = nn.Sequential(
                nn.Linear(hidden_dim * num_heads, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU(),
                nn.Linear(hidden_dim, num_link_type)
            )
            self.fc_node_type = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                nn.Linear(hidden_dim, len(feats_dim_list))
            )
            self.super_criterion = nn.CrossEntropyLoss()

    def forward(self, g: list, g_lists, features_list, type_mask,
                metapath_indices_list, target_idx_list, shift_list,
                link_indeices_list, link_counts, iden_g):
        # ntype-specific transformation
        features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            features[node_indices] = fc(features_list[i])
        features = self.feat_drop(features)

        if self.use_HMPNN:
            features = self.HGCN(g, {str(i): features for i in range(len(g[0].ntypes))})
            # features = self.HGCN(g, features, type_mask)

        if self.meta_mode in ['node_link', 'just_node']:
            # ctr_ntype-specific layers
            h_user, metapath_outs_user, _ = self.user_layer((g_lists[0], features, type_mask, metapath_indices_list[0],
                                                          target_idx_list[0]))
            h_item, metapath_outs_item, _ = self.item_layer((g_lists[1], features, type_mask, metapath_indices_list[1],
                                                          target_idx_list[1]))
            h_user = self.w_user(h_user)
            h_item = self.w_item(h_item)
            xij = h_user * h_item
            xij = self.fc_xij(xij)

        if self.meta_mode in ['node_link', 'just_link']:
            emb_links, metapath_link_outs, _ = self.link_layer((shift_list, features, type_mask, link_indeices_list))
            # emb_links = self.w_link(emb_links)
            emb_links = self.fc_link(emb_links)

        if self.meta_mode == 'node_link':
            out = self.lin(xij + emb_links * self.beta)
        elif self.meta_mode == 'just_node':
            out = self.lin(xij)
        elif self.meta_mode == 'just_link':
            out = self.lin(emb_links)

        out = torch.sigmoid(out)

        if self.self_super_meta:
            metapath_node_outs = torch.cat([metapath_outs_user, metapath_outs_item], dim=0)

            num_paths, batch_size, d_model = metapath_node_outs.shape
            assert batch_size == target_idx_list[0].shape[0]
            meta_node_label = torch.arange(num_paths).repeat_interleave(batch_size).to(metapath_node_outs.device)
            metapath_node_outs = metapath_node_outs.view(-1, d_model)  # (num_paths*batch_size, d_model)
            pred_node_meta = self.fc_node_meta_type(metapath_node_outs)
            loss_node = self.super_criterion(pred_node_meta, meta_node_label)

            num_paths, batch_size, d_model = metapath_link_outs.shape
            assert batch_size == target_idx_list[0].shape[0]
            meta_link_label = torch.arange(num_paths).repeat_interleave(batch_size).to(metapath_link_outs.device)
            metapath_link_outs = metapath_link_outs.view(-1, d_model)
            pred_link_meta = self.fc_link_meta_type(metapath_link_outs)
            loss_link = self.super_criterion(pred_link_meta, meta_link_label)

            node_label = torch.tensor(type_mask, dtype=torch.int64).to(features.device)
            pred_node_type = self.fc_node_type(features)
            loss_node_type = self.super_criterion(pred_node_type, node_label)

            return out.squeeze(1), loss_node + loss_link + loss_node_type
        else:
            return out.squeeze(1), None

    def fit(self, dl_train, dl_val, datas: tuple, param: tuple, early_stopping):
        net = self
        (g, adjlists_ua, edge_metapath_indices_list_ua, type_mask, features_list, adjM,
         adjlists_01, edge_metapath_indices_list_01, dl) = datas
        lr, weight_decay, num_epochs, device, neighbor_samples, use_masks, no_masks, logfile_path, _ = param

        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75, eta_min=8e-5)
        criterion = nn.BCELoss()
        net.train()

        for epoch in range(num_epochs):
            for batch_idx, (x_batch, y_batch) in enumerate(dl_train, start=1):
                net.train()

                (train_g_lists, train_indices_lists, target_idx_list,
                 shift_list, link_indeices_list, link_counts, iden_g) = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua,
                    adjlists_01, edge_metapath_indices_list_01, adjM,
                    x_batch.numpy(), device, neighbor_samples, use_masks)

                preds, loss_node_type = net(g, train_g_lists, features_list, type_mask,
                                             train_indices_lists, target_idx_list,
                                             shift_list, link_indeices_list, link_counts, iden_g)
                if self.self_super_meta:
                    train_loss = criterion(preds, y_batch.to(device)) + 0.5 * loss_node_type
                else:
                    train_loss = criterion(preds, y_batch.to(device))

                print('Epoch {:d} | Iteration {:d} | Train_Loss {:.5f} | lr {:.6f}'.format(epoch,
                                                                                           batch_idx,
                                                                                           train_loss.item(),
                                                                                           schedule.get_last_lr()[0]))

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                schedule.step()

                del train_g_lists, train_indices_lists, target_idx_list, shift_list, link_indeices_list, iden_g

                if batch_idx % 3 == 0 and batch_idx >= 6:
                # if False:
                    val_loss, result = self.evaluate(dl_val, datas, param)
                    auc, ap = result['roc_auc'], result['AP']
                    output_content = 'Epoch {:d} | Iteration {:d} | Train_Loss {:.5f} | Val_Loss {:.5f} | AUC {:.5f} | AP {:.5f}'.format(
                        epoch, batch_idx, train_loss.item(), val_loss, auc, ap)
                    print(output_content)
                    with open(logfile_path, "a", encoding="utf-8") as f:
                        print(output_content, file=f)
                    early_stopping(-auc, net)
                    if early_stopping.early_stop:
                        print('Early stopping!')
                        return

    def evaluate(self, dl_val, datas: tuple, param: tuple, savez=False):
        (g, adjlists_ua, edge_metapath_indices_list_ua, type_mask, features_list, adjM,
         adjlists_01, edge_metapath_indices_list_01, dl) = datas
        lr, weight_decay, num_epochs, device, neighbor_samples, use_masks, no_masks, logfile_path, p_data_path = param
        criterion = nn.BCELoss()
        net = self

        net.eval()
        running_loss = 0.0
        total_samples = 0
        y_true_val, y_proba_val = [], []
        x_all = []
        with torch.no_grad():
            print("\nevaluating...")
            for batch_idx, (x_batch, y_batch) in tqdm(enumerate(dl_val, start=1)):

                (val_g_lists, val_indices_lists, target_idx_list,
                 shift_list, link_indeices_list, link_counts, iden_g) = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua,
                    adjlists_01, edge_metapath_indices_list_01, adjM,
                    x_batch.numpy(), device, neighbor_samples*2, no_masks, test_mode=True)

                # print(f" num of edges {sum([g.number_of_edges() for g_lists in val_g_lists for g in g_lists])}")

                preds_val, _ = net(g, val_g_lists, features_list, type_mask,
                                    val_indices_lists, target_idx_list,
                                    shift_list, link_indeices_list, link_counts, iden_g)
                # print()
                # print(preds_val[:10])
                # print(y_batch[:10])
                # print(link_counts[:10])
                # return None, None, None
                del val_g_lists, val_indices_lists, target_idx_list, shift_list, link_indeices_list

                loss = criterion(preds_val, y_batch.to(device))
                running_loss += loss.item() * y_batch.shape[0]
                total_samples += y_batch.shape[0]
                y_true_val.append(y_batch.cpu().numpy())
                y_proba_val.append(preds_val.cpu().numpy())
                x_all.extend(x_batch.tolist())
            val_loss = running_loss / total_samples
            y_true_val, y_proba_val = np.concatenate(y_true_val), np.concatenate(y_proba_val)
            result = dl.evaluate(_, y_proba_val, y_true_val)
            # print(get_top_confident_correct_predictions(y_proba_val, y_true_val, x_all))
            if savez:   # only test
                import pandas as pd
                df = pd.DataFrame({'y_true': y_true_val, 'y_score': y_proba_val})
                df.to_csv(p_data_path, index=False)

        return val_loss, result

    def test(self, dl, batch_size, datas, param, test_edge_type=0):
        logfile_path = param[-2]
        assert isinstance(logfile_path, str)
        dataset = logfile_path.split("_")[-2]

        test_neigh, test_label = dl.get_test_neigh_w_random()
        test_user_artist = np.array(test_neigh[test_edge_type]).T
        test_label = np.array(test_label[test_edge_type])
        dl_test = DataLoader(TensorDataset(torch.from_numpy(test_user_artist).long(),
                                           torch.from_numpy(test_label).float()),
                                            batch_size=batch_size, shuffle=False)

        # # 钩子
        # extractor = FeatureExtractor_beta()
        # extractor.register_hook(self, "link_layer")  # 在目标层注册钩子
        # extractor_lin = FeatureExtractor_lin()
        # extractor_lin.register_hook(self, "lin")

        test_loss, result = self.evaluate(dl_test, datas, param, savez=True)

        # feature_array = np.concatenate(extractor.features, axis=0)
        # columns = [f'feat_{i}' for i in range(feature_array.shape[1])]
        # df = pd.DataFrame(data=feature_array, columns=columns)
        # df.to_csv(f'features_beta_CoupleGNN_{dataset}_NCNC.csv', index=False)
        # feature_array = np.concatenate(extractor_lin.features, axis=0)
        # columns = [f'feat_{i}' for i in range(feature_array.shape[1])]
        # df = pd.DataFrame(data=feature_array, columns=columns)
        # df.to_csv(f'features_CoupleGNN_{dataset}_NCNC.csv', index=False)
        # extractor.remove_hook()
        # # 钩子结束

        print()
        print('----------------------------------------------------------------')
        print('Link Prediction Test')
        print(result)

        with open(logfile_path, "a", encoding="utf-8") as f:
            print(file=f)
            print('----------------------------------------------------------------', file=f)
            print('Link Prediction Test', file=f)
            print(result, file=f)
        f.close()

    def discover_unknown_links(self, dl, batch_size, datas, param, test_edge_type=0,
                               node2id_file: str = "", output_file: str = ""):
        (g, adjlists_ua, edge_metapath_indices_list_ua, type_mask, features_list, adjM,
         adjlists_01, edge_metapath_indices_list_01, dl) = datas
        lr, weight_decay, num_epochs, device, neighbor_samples, use_masks, no_masks, logfile_path, p_data_path = param

        links = sort_ndarray(np.array(dl.unknown_link[test_edge_type]).T)
        dl_disc = DataLoader(TensorDataset(torch.from_numpy(links).long()), batch_size=batch_size, shuffle=False)

        net = self
        net.eval()
        total_samples = 0
        y_proba_val = []
        x_all = []
        with torch.no_grad():
            print("\nevaluating...")
            for batch_idx, [x_batch] in tqdm(enumerate(dl_disc, start=1)):
                (val_g_lists, val_indices_lists, target_idx_list,
                 shift_list, link_indeices_list, link_counts, iden_g) = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua,
                    adjlists_01, edge_metapath_indices_list_01, adjM,
                    x_batch.numpy(), device, neighbor_samples * 2, no_masks, test_mode=True)

                preds_val, _ = net(g, val_g_lists, features_list, type_mask,
                                   val_indices_lists, target_idx_list,
                                   shift_list, link_indeices_list, link_counts, iden_g)

                del val_g_lists, val_indices_lists, target_idx_list, shift_list, link_indeices_list

                total_samples += x_batch.shape[0]
                y_proba_val.append(preds_val.cpu().numpy())
                x_all.extend(x_batch.tolist())
            y_proba_val = np.concatenate(y_proba_val)

        df = pd.DataFrame({'link': x_all, 'prediction': y_proba_val})
        df_sorted = df.sort_values(by='prediction', ascending=False)
        df = df_sorted.head(50)

        if node2id_file == "":
            id_to_name = dl.nodes['name']
        else:
            node_df = pd.read_csv(node2id_file, header=None, names=['name', 'id'])
            id_to_name = dict(zip(node_df['id'], node_df['name']))
        # print(id_to_name)

        def get_drug_names(edge):
            source_id, _ = edge
            drug_name = id_to_name.get(str(source_id), 'Unknown')
            if drug_name == "Unknown":
                print(f"unknown drug {source_id}")
                raise ValueError
            return drug_name

        def get_target_names(edge):
            _, target_id = edge
            target_name = id_to_name.get(str(target_id), 'Unknown')
            if target_name == "Unknown":
                print(f"unknown target {target_name}")
                raise ValueError
            return target_name

        df['drug'] = df['link'].apply(get_drug_names)
        df['target'] = df['link'].apply(get_target_names)

        df.to_csv(output_file, index=False)

    def discover_specific_links(self, dl, batch_size, datas, param, test_edge_type=0,
                                link_discover_opath: str = "", output_file: str = ""):
        (g, adjlists_ua, edge_metapath_indices_list_ua, type_mask, features_list, adjM,
         adjlists_01, edge_metapath_indices_list_01, dl) = datas
        lr, weight_decay, num_epochs, device, neighbor_samples, use_masks, no_masks, logfile_path, p_data_path = param

        link_df = pd.read_csv(link_discover_opath, index_col=None, header=None)
        links = sort_ndarray(np.array(link_df))
        dl_disc = DataLoader(TensorDataset(torch.from_numpy(links).long()), batch_size=batch_size, shuffle=False)

        net = self
        net.eval()
        total_samples = 0
        y_proba_val = []
        x_all = []
        with torch.no_grad():
            print("\nevaluating...")
            for batch_idx, [x_batch] in tqdm(enumerate(dl_disc, start=1)):
                (val_g_lists, val_indices_lists, target_idx_list,
                 shift_list, link_indeices_list, link_counts, iden_g) = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua,
                    adjlists_01, edge_metapath_indices_list_01, adjM,
                    x_batch.numpy(), device, neighbor_samples * 2, no_masks, test_mode=True)

                preds_val, _ = net(g, val_g_lists, features_list, type_mask,
                                   val_indices_lists, target_idx_list,
                                   shift_list, link_indeices_list, link_counts, iden_g)

                del val_g_lists, val_indices_lists, target_idx_list, shift_list, link_indeices_list

                total_samples += x_batch.shape[0]
                y_proba_val.append(preds_val.cpu().numpy())
                x_all.extend(x_batch.tolist())
            y_proba_val = np.concatenate(y_proba_val)

        df = pd.DataFrame({'link': x_all, 'prediction': y_proba_val})

        id_to_name = dl.nodes['name']

        def get_miRNA_names(edge):
            source_id, _ = edge
            miRNA_name = id_to_name.get(source_id, 'Unknown')
            if miRNA_name == "Unknown":
                print(f"unknown drug {source_id}")
                raise ValueError
            return miRNA_name

        df['miRNA'] = df['link'].apply(get_miRNA_names)
        df.to_csv(output_file, index=False)


from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    def __init__(self):
        self.features = []
        self.hook = None

    @abstractmethod
    def hook_fn(self, module, input, output):
        pass  # 抽象方法无需实现，子类必须覆盖

    def register_hook(self, model, target_layer):
        layer = getattr(model, target_layer)
        self.hook = layer.register_forward_hook(self.hook_fn)

    def remove_hook(self):
        if self.hook:
            self.hook.remove()


class FeatureExtractor_beta(FeatureExtractor):
    def hook_fn(self, module, input, output):
        self.features.append(output[-1].cpu().detach().numpy())


class FeatureExtractor_lin(FeatureExtractor):
    def hook_fn(self, module, input, output):
        self.features.append(input[0].cpu().detach().numpy())


def get_top_confident_correct_predictions(y_proba_val, y_true_val, x_all):
    y_proba_val = np.array(y_proba_val)
    y_true_val = np.array(y_true_val)
    x_all = np.array(x_all)
    y_pred_val = (y_proba_val > 0.5).astype(int)
    correct_predictions = (y_pred_val == y_true_val)

    zero_class_mask = (y_true_val == 0) & correct_predictions
    ones_class_mask = (y_true_val == 1) & correct_predictions

    zero_class_samples = np.argsort(y_proba_val[zero_class_mask])[:5]
    one_class_samples = np.argsort(-y_proba_val[ones_class_mask])[:5]  # 负号是为了从高到低排序

    top_zero_confident = {
        'predicted': y_pred_val[zero_class_mask][zero_class_samples],
        'true': y_true_val[zero_class_mask][zero_class_samples],
        'input': x_all[zero_class_mask][zero_class_samples]
    }

    top_one_confident = {
        'predicted': y_pred_val[ones_class_mask][one_class_samples],
        'true': y_true_val[ones_class_mask][one_class_samples],
        'input': x_all[ones_class_mask][one_class_samples]
    }

    return top_zero_confident, top_one_confident
