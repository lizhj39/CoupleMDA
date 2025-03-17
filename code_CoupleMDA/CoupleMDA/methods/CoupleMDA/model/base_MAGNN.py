import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from methods.CoupleGNN.utils.pytorchtools import TransformerEncoder, mul_head_fc, mha_batch


class Link_metapath_specific(nn.Module):
    def __init__(self, etypes, out_dim, num_heads, r_vec, encode_type='linear'):
        super(Link_metapath_specific, self).__init__()
        self.etypes = etypes
        self.out_dim = out_dim
        self.num_heads = num_heads
        if encode_type not in ['linear', 'RotatE0']:
            raise NotImplementedError
        self.encode_type = encode_type
        self.r_vec = r_vec
        # self.rnn = TransformerEncoder(d_model=out_dim, nhead=4, num_encoder_layers=2, max_len=5)
        # self.attn = nn.MultiheadAttention(out_dim, num_heads=4, batch_first=True)
        # self.pos_encoder = nn.Embedding(5, out_dim)
        nh_dim = num_heads * out_dim
        self.w = nn.Linear(out_dim, nh_dim)
        self.ff = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4), nn.ELU(),
            nn.Linear(out_dim * 4, out_dim)
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.norm3 = nn.LayerNorm(out_dim)

        tmp = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        nn.init.xavier_normal_(tmp.data, gain=1.414)
        del tmp

    def forward(self, inputs):
        shift, features, type_mask, edge_metapath_indices = inputs

        edata = F.embedding(edge_metapath_indices, features)    # (E, num_nodes, hidden_dim)
        # position_ids = torch.arange(edata.shape[1], dtype=torch.long, device=features.device)
        # position_ids = position_ids.unsqueeze(0).expand(edata.shape[0], edata.shape[1])
        # edata = edata + self.pos_encoder(position_ids)
        # attention = mha_batch(self.attn, edata, small_bs=25000)
        # edata = self.norm1(edata + attention)
        if self.encode_type == 'linear':
            hidden = self.w(torch.mean(edata, dim=1))               # (E, num_heads * out_dim)
            eft = hidden.view(-1, self.num_heads, self.out_dim)     # (E, num_heads, out_dim)
        elif self.encode_type == 'RotatE0':
            r_vec = F.normalize(self.r_vec, p=2, dim=2)
            r_vec = torch.stack((r_vec, r_vec), dim=1)
            r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
            r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)  # etypes x out_dim/2 x 2
            edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
            final_r_vec = torch.zeros([edata.shape[1], self.out_dim // 2, 2], device=edata.device)
            final_r_vec[-1, :, 0] = 1
            for i in range(final_r_vec.shape[0] - 2, -1, -1):
                # consider None edge (symmetric relation)
                if self.etypes[i] is not None:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] - \
                                           final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1]
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] + \
                                           final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 0]
                else:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
            for i in range(edata.shape[1] - 1):
                temp1 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] - \
                        edata[:, i, :, 1].clone() * final_r_vec[i, :, 1]
                temp2 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] + \
                        edata[:, i, :, 1].clone() * final_r_vec[i, :, 0]
                edata[:, i, :, 0] = temp1
                edata[:, i, :, 1] = temp2
            edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
            eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  # E x num_heads x out_dim

        bs = len(shift) - 1  # batch size
        ret = torch.zeros((bs, self.num_heads, self.out_dim), device=eft.device)
        for i in range(bs):
            ret[i] = eft[shift[i]: shift[i + 1]].sum(dim=0)
        # ret = self.norm2(ret)
        # ret = self.norm3(ret + mul_head_fc(self.ff, ret))
        ret = F.elu(ret)
        return ret


class MAGNN_metapath_specific(nn.Module):
    def __init__(self,
                 etypes,
                 out_dim,
                 num_heads,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.5,
                 alpha=0.01,
                 use_minibatch=True,
                 attn_switch=False,
                 inner_attn=False):
        super(MAGNN_metapath_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.rnn_type = rnn_type
        self.etypes = etypes
        self.r_vec = r_vec
        self.use_minibatch = use_minibatch
        self.attn_switch = attn_switch
        self.inner_attn = inner_attn

        nh_dim = num_heads * out_dim
        self.fc = nn.Linear(out_dim, out_dim)
        self.ff = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4), nn.ELU(),
            nn.Linear(out_dim * 4, out_dim)
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        # rnn-like metapath instance aggregator
        # consider multiple attention heads
        if rnn_type == 'gru':
            self.rnn = nn.GRU(out_dim, num_heads * out_dim)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim)
        elif rnn_type == 'bi-gru':
            self.rnn = nn.GRU(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'bi-lstm':
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'linear':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif rnn_type == 'max-pooling':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif rnn_type == 'neighbor-linear':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)

        # node-level attention
        # attention considers the center node embedding or not
        if self.inner_attn:
            if self.attn_switch:
                self.attn1 = nn.Linear(out_dim, num_heads, bias=False)
                self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
            else:
                self.attn = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
            self.leaky_relu = nn.LeakyReLU(alpha)
            if attn_drop:
                self.attn_drop = nn.Dropout(attn_drop)
            else:
                self.attn_drop = lambda x: x

            # weight initialization
            if self.attn_switch:
                nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
                nn.init.xavier_normal_(self.attn2.data, gain=1.414)
            else:
                nn.init.xavier_normal_(self.attn.data, gain=1.414)
        else:
            # the codes below do work
            tmp = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
            nn.init.xavier_normal_(tmp.data, gain=1.414)
            del tmp

    def message_passing(self, edges):
        if self.inner_attn:
            ft = edges.data['eft'] * edges.data['a_drop']
        else:
            ft = edges.data['eft']
        return {'ft': ft}

    def forward(self, inputs):
        # features: num_all_nodes x out_dim
        if self.use_minibatch:
            g, features, type_mask, edge_metapath_indices, target_idx = inputs
        else:
            g, features, type_mask, edge_metapath_indices = inputs

        # Embedding layer
        # use torch.nn.functional.embedding or torch.embedding here
        # do not use torch.nn.embedding
        # edata: E x Seq x out_dim
        edata = F.embedding(edge_metapath_indices, features)

        # apply rnn to metapath-based feature sequence
        if self.rnn_type == 'gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'bi-gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(0, 2, 1).reshape(
                -1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'bi-lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(0, 2, 1).reshape(
                -1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'average':
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'linear':
            hidden = self.rnn(torch.mean(edata, dim=1))
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'max-pooling':
            hidden, _ = torch.max(self.rnn(edata), dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'TransE0' or self.rnn_type == 'TransE1':
            r_vec = self.r_vec
            if self.rnn_type == 'TransE0':
                r_vec = torch.stack((r_vec, -r_vec), dim=1)
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1])  # etypes x out_dim
            edata = F.normalize(edata, p=2, dim=2)
            for i in range(edata.shape[1] - 1):
                # consider None edge (symmetric relation)
                temp_etypes = [etype for etype in self.etypes[i:] if etype is not None]
                edata[:, i] = edata[:, i] + r_vec[temp_etypes].sum(dim=0)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'RotatE0' or self.rnn_type == 'RotatE1':
            r_vec = F.normalize(self.r_vec, p=2, dim=2)
            if self.rnn_type == 'RotatE0':
                r_vec = torch.stack((r_vec, r_vec), dim=1)
                r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)  # etypes x out_dim/2 x 2
            edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
            final_r_vec = torch.zeros([edata.shape[1], self.out_dim // 2, 2], device=edata.device)
            final_r_vec[-1, :, 0] = 1
            for i in range(final_r_vec.shape[0] - 2, -1, -1):
                # consider None edge (symmetric relation)
                if self.etypes[i] is not None:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] -\
                                           final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1]
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] +\
                                           final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 0]
                else:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
            for i in range(edata.shape[1] - 1):
                temp1 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] -\
                        edata[:, i, :, 1].clone() * final_r_vec[i, :, 1]
                temp2 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] +\
                        edata[:, i, :, 1].clone() * final_r_vec[i, :, 0]
                edata[:, i, :, 0] = temp1
                edata[:, i, :, 1] = temp2
            edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'neighbor':
            hidden = edata[:, 0]
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'neighbor-linear':
            hidden = self.rnn(edata[:, 0])
            hidden = hidden.unsqueeze(dim=0)

        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  # E x num_heads x out_dim
        g.edata.update({'eft': eft})
        if self.inner_attn:
            if self.attn_switch:
                center_node_feat = F.embedding(edge_metapath_indices[:, -1], features)  # E x out_dim
                a1 = self.attn1(center_node_feat)  # E x num_heads
                a2 = (eft * self.attn2).sum(dim=-1)  # E x num_heads
                a = (a1 + a2).unsqueeze(dim=-1)  # E x num_heads x 1
            else:
                a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # E x num_heads x 1
            a = self.leaky_relu(a)
            g.edata['a_drop'] = self.attn_drop(edge_softmax(g, a))

        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        ret = g.ndata['ft']  # N x num_heads x out_dim

        if self.use_minibatch:
            ret = ret[target_idx]

        # ret = self.norm1(ret)
        # ret = self.norm2(ret + mul_head_fc(self.ff, ret))
        ret = F.elu(ret)
        return ret


class MAGNN_ctr_ntype_specific(nn.Module):
    def __init__(self,
                 num_metapaths,
                 etypes_list,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 l_enco_type='RotatE0',
                 r_vec=None,
                 attn_drop=0.5,
                 use_minibatch=True,
                 link_mode=False,
                 use_self_attn=True,
                 inner_attn=False,
                 self_super_meta=False):
        super(MAGNN_ctr_ntype_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch
        self.link_mode = link_mode
        self.use_self_attn = use_self_attn
        self.inner_attn = inner_attn
        self.self_super_meta = self_super_meta

        nh_dim = num_heads * out_dim
        self.fc = nn.Linear(nh_dim, nh_dim)
        self.ff = nn.Sequential(
            nn.Linear(nh_dim, nh_dim * 2), nn.ELU(),
            nn.Linear(nh_dim * 2, nh_dim)
        )
        self.norm1 = nn.LayerNorm(nh_dim)
        self.norm2 = nn.LayerNorm(nh_dim)

        # metapath-specific layers
        self.metapath_layers = nn.ModuleList()
        if self.link_mode:
            for i in range(num_metapaths):
                self.metapath_layers.append(Link_metapath_specific(etypes_list[i], out_dim, num_heads,
                                                                   r_vec=r_vec, encode_type=l_enco_type))
        else:
            for i in range(num_metapaths):
                self.metapath_layers.append(MAGNN_metapath_specific(etypes_list[i],
                                                                    out_dim,
                                                                    num_heads,
                                                                    rnn_type,
                                                                    r_vec,
                                                                    attn_drop=attn_drop,
                                                                    use_minibatch=use_minibatch,
                                                                    inner_attn=self.inner_attn))

        # metapath-level attention
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)
        if self.use_self_attn:
            self.attention = nn.MultiheadAttention(embed_dim=attn_vec_dim, num_heads=4, batch_first=True)

        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        if self.link_mode:
            shift_list, features, type_mask, edge_metapath_indices_list = inputs
            metapath_outs = [
                metapath_layer(
                    (shift, features, type_mask, edge_metapath_indices)).view(-1, self.num_heads * self.out_dim)
                for shift, edge_metapath_indices, metapath_layer in
                zip(shift_list, edge_metapath_indices_list, self.metapath_layers)]
        else:
            if self.use_minibatch:
                g_list, features, type_mask, edge_metapath_indices_list, target_idx_array = inputs

                # metapath-specific layers
                metapath_outs = [
                    metapath_layer(
                        (g, features, type_mask, edge_metapath_indices, target_idx_array)).view(-1, self.num_heads * self.out_dim)
                    for g, edge_metapath_indices, metapath_layer in
                    zip(g_list, edge_metapath_indices_list, self.metapath_layers)]
            else:
                g_list, features, type_mask, edge_metapath_indices_list = inputs

                # metapath-specific layers
                metapath_outs = [
                    metapath_layer(
                        (g, features, type_mask, edge_metapath_indices)).view(-1, self.num_heads * self.out_dim)
                    for g, edge_metapath_indices, metapath_layer in
                    zip(g_list, edge_metapath_indices_list, self.metapath_layers)]

        metapath_outs = torch.stack(metapath_outs, dim=0).transpose(0, 1)  # (batch_size, num_paths, out_dim*num_heads)

        if self.use_self_attn:
            fc1 = torch.tanh(self.fc1(metapath_outs))   # (batch_size, num_paths, attn_vec_dim)
            attn_output, attn_weights = self.attention(fc1, fc1, fc1)   # (batch_size, num_paths, attn_vec_dim), (batch_size, num_paths, num_paths)
            fc2 = self.fc2(attn_output).squeeze(-1)  # (batch_size, num_paths)
            beta = F.softmax(fc2, dim=1)             # (batch_size, num_paths)
            h = torch.sum(beta.unsqueeze(-1) * metapath_outs, dim=1)    # (batch_size, num_paths, 1) * (batch_size, num_paths, out_dim * num_heads)
        else:
            fc1 = torch.tanh(self.fc1(metapath_outs))       # (batch_size, num_paths, attn_vec_dim)
            fc1_mean = torch.mean(fc1, dim=0)               # (num_paths, attn_vec_dim)
            fc2 = self.fc2(fc1_mean)                        # (num_paths, 1)
            beta = F.softmax(fc2, dim=0)
            beta = torch.unsqueeze(beta, dim=-1)            # (num_paths, 1, 1)
            metapath_outs_copy = metapath_outs.transpose(0, 1)   # (num_paths, batch_size, out_dim*num_heads)
            h = torch.sum(beta * metapath_outs_copy, dim=0)

        # h = self.norm1(h + self.fc(h))
        # h = self.norm2(h + self.ff(h))

        if self.self_super_meta:
            return h, metapath_outs.transpose(0, 1), beta     # metapath_outs: (num_paths, batch_size, out_dim*num_heads)
        else:
            return h, None, beta  # (batch_size, out_dim * num_heads)
