import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch import HeteroGraphConv, GraphConv, SAGEConv, HGTConv
from torch.nn import LayerNorm, Linear


class OrderedGCN(nn.Module):
    def __init__(self, conv, f_shift: list, hidden_channel=64, chunk_size=32):
        super(OrderedGCN, self).__init__()
        self.conv = conv
        self.f_shift = f_shift
        self.tm_net = Linear(2 * hidden_channel, chunk_size)
        self.tm_norm = LayerNorm(hidden_channel)
        self.hidden_channel = hidden_channel
        self.chunk_size = chunk_size

    def forward(self, g, x_dict: dict, last_tm_signal):
        m_dict = self.conv(g, x_dict)
        m = get_tensor(m_dict, self.f_shift)
        x = get_tensor(x_dict, self.f_shift)

        tm_signal_raw = F.softmax(self.tm_net(torch.cat((x, m), dim=1)), dim=-1)
        tm_signal_raw = torch.cumsum(tm_signal_raw, dim=-1)
        tm_signal_raw = last_tm_signal + (1 - last_tm_signal) * tm_signal_raw
        tm_signal = tm_signal_raw.repeat_interleave(repeats=int(self.hidden_channel / self.chunk_size), dim=1)

        out = x * tm_signal + m * (1 - tm_signal)
        out = self.tm_norm(out)
        return out, tm_signal_raw


class GatedGCNLayer(nn.Module):
    def __init__(self, conv, f_shift, in_feats, out_feats):
        super(GatedGCNLayer, self).__init__()
        self.conv = conv
        self.f_shift = f_shift
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.gate_fc = nn.Linear(in_feats, out_feats)

    def forward(self, g, h_dict: dict):
        with g.local_scope():
            agg_h_dict = self.conv(g, h_dict)
            agg_h = get_tensor(agg_h_dict, self.f_shift)

            h = get_tensor(h_dict, self.f_shift)
            gate = torch.sigmoid(self.gate_fc(h))

            gated_h = gate * agg_h + (1 - gate) * h  # 结合原始特征和卷积特征
            return gated_h


class HeteroGNN(nn.Module):
    def __init__(self, in_feats, out_feats, edge_types: list, f_shift: list,
                 num_layers=2, hidden_channel=64, chunk_size=32):
        super(HeteroGNN, self).__init__()
        self.f_shift = f_shift
        self.chunk_size = chunk_size
        self.num_etypes = len(edge_types)

        self.conv1 = HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types})
        # self.conv2 = HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types})
        # self.conv3 = HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types})
        # self.conv4 = HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types})
        # self.conv5 = HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types})

        # self.conv1 = GatedGCNLayer(conv=HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types}),
        #                            f_shift=f_shift, in_feats=in_feats, out_feats=out_feats)
        # self.conv2 = GatedGCNLayer(conv=HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types}),
        #                            f_shift=f_shift, in_feats=in_feats, out_feats=out_feats)
        # self.conv3 = GatedGCNLayer(conv=HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types}),
        #                            f_shift=f_shift, in_feats=in_feats, out_feats=out_feats)
        # self.conv4 = GatedGCNLayer(conv=HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types}),
        #                            f_shift=f_shift, in_feats=in_feats, out_feats=out_feats)
        # self.conv5 = GatedGCNLayer(conv=HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types}),
        #                            f_shift=f_shift, in_feats=in_feats, out_feats=out_feats)

        # self.conv1 = OrderedGCN(conv=HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types}),
        #                         f_shift=f_shift)
        # self.conv2 = OrderedGCN(conv=HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types}),
        #                         f_shift=f_shift)
        # self.conv3 = OrderedGCN(conv=HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types}),
        #                         f_shift=f_shift)
        # self.conv4 = OrderedGCN(conv=HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types}),
        #                         f_shift=f_shift)
        # self.conv5 = OrderedGCN(conv=HeteroGraphConv({e_type: GraphConv(in_feats, out_feats) for e_type in edge_types}),
        #                         f_shift=f_shift)


    def forward(self, g: list, h_dict: dict):
        g_ori, g_selfloop = g[0], g[1]

        h_dict = self.conv1(g_ori, h_dict)
        # h_dict = {k: F.elu(v) for k, v in h_dict.items()}
        # h_dict = self.conv2(g_selfloop, h_dict)
        # h_dict = {k: F.elu(v) for k, v in h_dict.items()}
        # h_dict = self.conv3(g_selfloop, h_dict)
        # h_dict = {k: F.elu(v) for k, v in h_dict.items()}
        # h_dict = self.conv4(g_selfloop, h_dict)
        # h_dict = {k: F.elu(v) for k, v in h_dict.items()}
        # h_dict = self.conv5(g_selfloop, h_dict)
        h = get_tensor(h_dict, self.f_shift)

        # h = self.conv1(g_ori, h_dict)
        # h = F.elu(h)
        # h_dict = tensor2dict(h, len(h_dict))
        # h = self.conv2(g_selfloop, h_dict)
        # h = F.elu(h)
        # h_dict = tensor2dict(h, len(h_dict))
        # h = self.conv3(g_selfloop, h_dict)
        # h = F.elu(h)
        # h_dict = tensor2dict(h, len(h_dict))
        # h = self.conv4(g_selfloop, h_dict)
        # h = F.elu(h)
        # h_dict = tensor2dict(h, len(h_dict))
        # h = self.conv5(g_selfloop, h_dict)

        # tm_signal = h_dict['0'].new_zeros(self.chunk_size)
        # h, tm_signal = self.conv1(g_ori, h_dict, last_tm_signal=tm_signal)
        # h_dict = tensor2dict(h, len(h_dict))
        # h, tm_signal = self.conv2(g_selfloop, h_dict, last_tm_signal=tm_signal)
        # h_dict = tensor2dict(h, len(h_dict))
        # h, tm_signal = self.conv3(g_selfloop, h_dict, last_tm_signal=tm_signal)
        # h_dict = tensor2dict(h, len(h_dict))
        # h, tm_signal = self.conv4(g_selfloop, h_dict, last_tm_signal=tm_signal)
        # h_dict = tensor2dict(h, len(h_dict))
        # h, tm_signal = self.conv5(g_selfloop, h_dict, last_tm_signal=tm_signal)

        return h


class HGT(nn.Module):
    def __init__(self, in_feats, out_feats, edge_types: list, f_shift: list, num_heads=4):
        super(HGT, self).__init__()
        self.conv1 = HGTConv(in_feats, out_feats//num_heads,
                             num_heads=num_heads,
                             num_ntypes=3, num_etypes=6)
        self.conv2 = HGTConv(in_feats, out_feats // num_heads,
                             num_heads=num_heads,
                             num_ntypes=3, num_etypes=6)
        self.conv3 = HGTConv(in_feats, out_feats // num_heads,
                             num_heads=num_heads,
                             num_ntypes=3, num_etypes=6)
        self.conv4 = HGTConv(in_feats, out_feats // num_heads,
                             num_heads=num_heads,
                             num_ntypes=3, num_etypes=6)
        self.conv5 = HGTConv(in_feats, out_feats // num_heads,
                             num_heads=num_heads,
                             num_ntypes=3, num_etypes=6)

    def forward(self, g: list, x: torch.Tensor, type_mask: list):
        g_homo, etypes = g[2], g[3]
        device = g_homo.device

        x = self.conv1(g_homo, x, ntype=torch.tensor(type_mask).to(device), etype=torch.tensor(etypes).to(device))
        x = self.conv2(g_homo, x, ntype=torch.tensor(type_mask).to(device), etype=torch.tensor(etypes).to(device))
        x = self.conv3(g_homo, x, ntype=torch.tensor(type_mask).to(device), etype=torch.tensor(etypes).to(device))
        x = self.conv4(g_homo, x, ntype=torch.tensor(type_mask).to(device), etype=torch.tensor(etypes).to(device))
        x = self.conv5(g_homo, x, ntype=torch.tensor(type_mask).to(device), etype=torch.tensor(etypes).to(device))
        return x


def get_tensor(h_dict: dict, f_shift: list):
    h_tensor_list = list(h_dict.values())
    h_tensor = torch.cat([h[f_shift[i]: f_shift[i+1]] for i, h in enumerate(h_tensor_list)], dim=0)
    return h_tensor


def tensor2dict(h_tensor, ntypes):
    return {str(i): h_tensor for i in range(ntypes)}
