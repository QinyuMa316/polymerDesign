import torch
import torch.nn as nn
import torch.nn.functional as F

# 对比学习损失函数
# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.1):
#         super(ContrastiveLoss, self).__init__()
#         self.temperature = temperature
#         self.criterion = nn.CrossEntropyLoss()
#
#     def forward(self, projections):
#         # 计算相似度矩阵
#         sim_matrix = torch.mm(projections, projections.t()) / self.temperature
#         # 创建标签: 同一分子的不同视图为正样本
#         labels = torch.arange(projections.size(0), device=projections.device)
#         labels = labels - labels % 2  # 将相邻的两个样本视为同一分子
#         return self.criterion(sim_matrix, labels)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        # 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.size(0)
        # 计算相似度矩阵
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature

        # 对角线是正样本对
        labels = torch.arange(batch_size).to(z1.device)

        loss = (self.criterion(sim_matrix, labels) + self.criterion(sim_matrix.T, labels)) / 2
        return loss


