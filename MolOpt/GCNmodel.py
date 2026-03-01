
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, global_mean_pool  # 添加NNConv支持边特征
import warnings

warnings.filterwarnings("ignore")


class GCNModel(nn.Module):
    def __init__(self, num_node_features=7, num_edge_features=6, hidden_dim=64,
                 num_heads=4, dropout=0.2, num_layers=3):
        super(GCNModel, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # 定义边特征处理网络
        edge_network1 = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_node_features * hidden_dim)
        )

        edge_network_mid = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )

        edge_network_last = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )

        # 第一层GCN：使用NNConv整合边特征
        self.conv1 = NNConv(num_node_features, hidden_dim,
                            nn=edge_network1, aggr='mean')  # 使用NNConv

        # 中间层GCN
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            edge_network = nn.Sequential(
                nn.Linear(num_edge_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim)
            )
            self.convs.append(
                NNConv(hidden_dim, hidden_dim,
                       nn=edge_network_mid, aggr='mean')  # 使用NNConv
            )

        # 最后一层GCN
        self.conv_last = NNConv(hidden_dim, hidden_dim,
                                nn=edge_network_last, aggr='mean')  # 使用NNConv
        self.hidden_dim = hidden_dim

        # 全连接层保持不变
        self.mlp_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.mlp_proj_cl = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, data, return_cl_dim=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 第一层：整合边特征
        x = F.elu(self.conv1(x, edge_index, edge_attr))  # 添加edge_attr
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 中间层：整合边特征
        for conv in self.convs:
            x = F.elu(conv(x, edge_index, edge_attr))  # 添加edge_attr
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层：整合边特征
        x = F.elu(self.conv_last(x, edge_index, edge_attr))  # 添加edge_attr

        readout = global_mean_pool(x, batch)
        if return_cl_dim:
            out = self.mlp_proj_cl(readout)
            return out, readout
        else:
            out = self.mlp_proj(readout)
            out = out.squeeze(-1)
            return out, readout

class GCNModel0(nn.Module):  # 类名更新为GCNModel
    def __init__(self, num_node_features=7, num_edge_features=6, hidden_dim=64,
                 num_heads=4, dropout=0.2, num_layers=3):
        super(GCNModel0, self).__init__()  # 类名更新

        self.num_layers = num_layers
        self.dropout = dropout

        # 第一层GCN：忽略edge_attr
        self.conv1 = GCNConv(num_node_features, hidden_dim)  # 替换为GCNConv

        # 中间层GCN
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_dim, hidden_dim)  # 替换为GCNConv，移除多头
            )

        # 最后一层GCN
        self.conv_last = GCNConv(hidden_dim, hidden_dim)  # 替换为GCNConv
        self.hidden_dim = hidden_dim

        # 全连接层保持不变
        self.mlp_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.mlp_proj_cl = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, data, return_cl_dim=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 第一层：忽略edge_attr
        x = F.elu(self.conv1(x, edge_index))  # 移除edge_attr
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 中间层：忽略edge_attr
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))  # 移除edge_attr
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层：忽略edge_attr
        x = F.elu(self.conv_last(x, edge_index))  # 移除edge_attr

        readout = global_mean_pool(x, batch)
        if return_cl_dim:
            out = self.mlp_proj_cl(readout)
            return out, readout
        else:
            out = self.mlp_proj(readout)
            out = out.squeeze(-1)
            return out, readout




