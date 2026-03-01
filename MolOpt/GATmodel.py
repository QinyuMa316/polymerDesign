import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import warnings
warnings.filterwarnings("ignore")

class GATModel(nn.Module):
    def __init__(self, num_node_features=7, num_edge_features=6, hidden_dim=64,
                 num_heads=4, dropout=0.2, num_layers=3):
        super(GATModel, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        # 第一层GAT
        self.conv1 = GATConv(in_channels = num_node_features,
                             out_channels = hidden_dim,
                             heads = num_heads,
                             dropout = dropout,
                             edge_dim = num_edge_features,
                             concat = True)
        # 中间层GAT
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(in_channels = hidden_dim * num_heads,
                        out_channels = hidden_dim,
                        heads = num_heads,
                        dropout = dropout,
                        edge_dim = num_edge_features,
                        concat=True)
            )
        # 最后一层GAT
        self.conv_last = GATConv(in_channels = hidden_dim * num_heads,
                                 out_channels = hidden_dim,
                                 heads=1,
                                 dropout=dropout,
                                 edge_dim=num_edge_features,
                                 concat=False)
        self.hidden_dim = hidden_dim
        # 全连接层（监督学习使用）
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

    def forward(self, data, return_cl_dim = False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 第一层
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 中间层
        for conv in self.convs:
            x = F.elu(conv(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层
        x = F.elu(self.conv_last(x, edge_index, edge_attr))

        readout = global_mean_pool(x, batch)
        if return_cl_dim:
            out = self.mlp_proj_cl(readout)
            return out, readout
        else:
            out = self.mlp_proj(readout)
            out = out.squeeze(-1)
            return out, readout
