import torch
from torch_geometric.data import Data, Dataset, DataLoader
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class MolDataset(Dataset):
    def __init__(self, smiles_list, label_list,
                 transform=None,
                 contrastive = False,
                 ):
        self.smiles_list = smiles_list
        self.label_list = label_list
        self.transform = transform
        self.contrastive = contrastive

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):

        smiles = self.smiles_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 获取原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.atom_feature(atom))
        x = torch.tensor(atom_features, dtype=torch.float)

        # 获取边索引和边特征
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append((i, j))
            edge_index.append((j, i))
            edge_attr.append(self.bond_feature(bond))
            edge_attr.append(self.bond_feature(bond))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # label = self.label_list[idx]
        # y = torch.tensor([label], dtype=torch.float)
        # data_mod = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)
        # return data_mod

        if self.contrastive:
            # 对比学习模式：返回两个增强视图
            view1 = self._create_view12(mol)
            view2 = self._create_view12(mol)
            return view1, view2
        else:
            # 监督学习模式
            label = self.label_list[idx]
            y = torch.tensor([label], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)
            # if self.transform:
            #     data_mod = self.transform(data_mod)
            return data

    def atom_feature(self, atom):
        # 定义原子特征，如原子序数、杂化状态、是否在环中等
        return [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization().real,
            atom.GetIsAromatic(),
            atom.IsInRing()
        ]

    def bond_feature(self, bond):
        # 定义键特征，如键类型、是否共轭、是否在环中等
        bt = bond.GetBondType()
        return [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]

    def _create_view1(self, mol):
        """创建增强视图"""
        # 简化版：使用不同的随机SMILES表示

        # 使用随机SMILES创建增强视图
        smiles = Chem.MolToSmiles(mol, doRandom=True)
        mol = Chem.MolFromSmiles(smiles)

        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.atom_feature(atom))
        x = torch.tensor(atom_features, dtype=torch.float)

        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append((i, j))
            edge_index.append((j, i))
            edge_attr.append(self.bond_feature(bond))
            edge_attr.append(self.bond_feature(bond))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _create_view2(self, mol):
        """创建增强视图"""
        # 随机选择一种增强策略
        strategy = np.random.choice(['atom_drop', 'edge_drop', 'atom_mask', 'edge_mask'])

        # 复制原始分子
        mol = Chem.RWMol(mol)

        if strategy == 'atom_drop' and mol.GetNumAtoms() > 1:
            # 策略1: 随机删除5%原子
            n_drop = max(1, int(mol.GetNumAtoms() * 0.05))  # 至少删除1个原子
            atom_indices = list(range(mol.GetNumAtoms()))
            to_drop = np.random.choice(atom_indices, size=n_drop, replace=False)
            # 从大到小排序，避免删除时索引变化
            for idx in sorted(to_drop, reverse=True):
                mol.RemoveAtom(int(idx))  # 确保转换为Python原生int

        elif strategy == 'edge_drop' and mol.GetNumBonds() > 1:
            # 策略2: 随机删除5%边
            n_drop = max(1, int(mol.GetNumBonds() * 0.05))  # 至少删除1条边
            bond_indices = list(range(mol.GetNumBonds()))
            to_drop = np.random.choice(bond_indices, size=n_drop, replace=False)
            # 从大到小排序，避免删除时索引变化
            for idx in sorted(to_drop, reverse=True):
                bond = mol.GetBondWithIdx(int(idx))  # 确保转换为Python原生int
                begin_atom = bond.GetBeginAtomIdx()
                end_atom = bond.GetEndAtomIdx()
                mol.RemoveBond(begin_atom, end_atom)

        # 转换为普通分子
        mol = mol.GetMol()

        # 获取原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            features = self.atom_feature(atom)
            if strategy == 'atom_mask' and np.random.rand() < 0.05:
                # 策略3: 随机mask 5%原子特征
                features = [0 if np.random.rand() < 0.5 else f for f in features]
            atom_features.append(features)
        x = torch.tensor(atom_features, dtype=torch.float)

        # 获取边索引和边特征
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append((i, j))
            edge_index.append((j, i))
            features = self.bond_feature(bond)
            if strategy == 'edge_mask' and np.random.rand() < 0.05:
                # 策略4: 随机mask 5%边特征
                features = [0 if np.random.rand() < 0.5 else f for f in features]
            edge_attr.append(features)
            edge_attr.append(features)  # 双向边

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _create_view12(self, mol):
        """
        先执行 _create_view (随机 SMILES)，
        再把得到的随机 SMILES 结果交给 _create_view2 做结构级增强，
        从而在同一视图里叠加两种随机性。
        """
        # 第一步：随机 SMILES → 得到一个新的 Mol
        rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
        rand_mol    = Chem.MolFromSmiles(rand_smiles)
        # 第二步：在新的 Mol 上做 drop/mask 等增强
        return self._create_view2(rand_mol)

class MolDataLoader:
    def __init__(self, smiles_list, label_list, batch_size=32, shuffle=True):
        """
        初始化 MolDataLoader
        :param smiles_list: SMILES 字符串列表
        :param label_list: 对应的标签列表
        :param batch_size: 批量大小
        :param test_size: 测试集比例
        :param val_size: 验证集比例
        :param shuffle: 是否对数据进行随机打乱
        """
        assert len(smiles_list) == len(label_list), "SMILES 列表与标签列表长度必须相等"
        self.smiles_list = smiles_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_all_loader(self):
        dataset = MolDataset(self.smiles_list, self.label_list, contrastive=False)
        all_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return all_loader

    def get_split_loaders_cl(self, test_size=0.1, val_size=0.1):
        """
        获取数据加载器
        :return: 训练、验证、测试集的 DataLoader
        """

        # ======== 1) 如果 test_size > 0，则进行测试集拆分，否则留空集 ========
        if test_size > 0:
            smiles_train, smiles_test, labels_train, labels_test = train_test_split(
                self.smiles_list, self.label_list, test_size=test_size,
                random_state=42, shuffle=self.shuffle
            )
        else:
            # 不拆分
            smiles_train = self.smiles_list
            labels_train = self.label_list
            smiles_test = []
            labels_test = []

        # ======== 2) 如果 val_size > 0，则从 train 中进一步拆分验证集，否则留空集 ========
        if val_size > 0:
            # 这里为了防止分母=0，如果 test_size=1，val_size/(1-test_size)会报错
            # 一般不会出现 test_size=1 的极端情况，但可做一下保护
            remain_ratio = 1 - test_size
            if remain_ratio <= 0:
                raise ValueError("test_size=1, 没有数据可给验证集，请调整 test_size 或 val_size")

            smiles_train, smiles_val, labels_train, labels_val = train_test_split(
                smiles_train, labels_train,
                test_size = val_size / remain_ratio,
                random_state = 42, shuffle=self.shuffle
            )
        else:
            # 不拆分
            smiles_val = []
            labels_val = []
        ###

        # 创建数据集
        self.train_dataset = MolDataset(smiles_train, labels_train, contrastive=True)
        self.val_dataset = MolDataset(smiles_val, labels_val, contrastive=True)
        self.test_dataset = MolDataset(smiles_test, labels_test, contrastive=True)

        from torch_geometric.data import Batch

        def contrastive_collate(batch):
            view1 = [data[0] for data in batch]
            view2 = [data[1] for data in batch]
            return Batch.from_data_list(view1), Batch.from_data_list(view2)

        # 创建 DataLoader
        # train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=contrastive_collate  # 添加此行
        )
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def enumerate_smiles(self, smi: str, n: int = 8):
        """
        给定 canonical SMILES，返回 n 条随机 SMILES（含自身），去重
        """
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return []  # 无法解析的直接跳过
        rand_set = {Chem.MolToSmiles(mol, doRandom=True) for _ in range(n)}
        rand_set.add(Chem.MolToSmiles(mol, canonical=True))  # 保障含原始
        return list(rand_set)

    def _augment(self, S, L, n_enum):
        aug_s, aug_l = [], []
        for s, lab in zip(S, L):
            for a in self.enumerate_smiles(s, n_enum):
                aug_s.append(a)
                aug_l.append(lab)
        return aug_s, aug_l

    def get_split_loaders(self, test_size=0.1, val_size=0.1, n_enum = 8, random_state = 42): # === modified ===
        """
        返回 train/val/test 的 DataLoader
        先按 canonical SMILES 分组拆分，再在每个 split 内做随机 SMILES 枚举
        """
        # ===== 1) 先按“分子”分组划分 train/test =====
        groups = [smi for smi in self.smiles_list] # 每个 canonical SMILES 自成一组
        # gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        # train_idx, test_idx = next(gss.split(self.smiles_list, groups=groups))
        if test_size == 0:  # === 新增：允许 test_size 为 0
            train_idx = np.arange(len(self.smiles_list))
            test_idx = np.array([], dtype=int)
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(self.smiles_list, groups=groups))

        smiles_train = [self.smiles_list[i] for i in train_idx]
        labels_train = [self.label_list[i]  for i in train_idx]
        smiles_test  = [self.smiles_list[i] for i in test_idx]
        labels_test  = [self.label_list[i]  for i in test_idx]

        # ===== 2) 从 train 中再拆分 val =====
        if val_size > 0:
            gss2 = GroupShuffleSplit(n_splits=1,
                                     test_size=val_size / (1 - test_size),
                                     random_state=random_state)
            tr_idx, val_idx = next(gss2.split(smiles_train, groups=smiles_train))
            smiles_val = [smiles_train[i] for i in val_idx]
            labels_val = [labels_train[i] for i in val_idx]
            smiles_train = [smiles_train[i] for i in tr_idx]
            labels_train = [labels_train[i] for i in tr_idx]
        else:
            smiles_val, labels_val = [], []

        # ===== 3) split 内部做随机 SMILES 枚举 =====
        if n_enum > 0:
            smiles_train, labels_train = self._augment(smiles_train, labels_train, n_enum)
            smiles_val, labels_val = self._augment(smiles_val, labels_val, n_enum)
            smiles_test, labels_test = self._augment(smiles_test, labels_test, n_enum)

        # ===== 4) 构建数据集 & DataLoader =====
        self.train_dataset = MolDataset(smiles_train, labels_train, contrastive=False)
        self.val_dataset = MolDataset(smiles_val, labels_val, contrastive=False)
        self.test_dataset = MolDataset(smiles_test, labels_test, contrastive=False)

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        # note: val & test shuffle == false
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

#
# from sklearn.model_selection import KFold
#
# class KFoldMolDataLoader:
#     def __init__(self, smiles_list, label_list,
#                  n_splits,
#                  batch_size=32, shuffle=True):
#         self.smiles_list = smiles_list
#         self.label_list = label_list
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.n_splits = n_splits
#         self.kf = KFold(n_splits=n_splits, shuffle=shuffle)
#
#     def enumerate_smiles(self, smi: str, n: int = 8):
#         """
#         给定 canonical SMILES，返回 n 条随机 SMILES（含自身），去重
#         """
#         mol = Chem.MolFromSmiles(smi)
#         if mol is None:
#             return []  # 无法解析的直接跳过
#         rand_set = {Chem.MolToSmiles(mol, doRandom=True) for _ in range(n)}
#         rand_set.add(Chem.MolToSmiles(mol, canonical=True))  # 保障含原始
#         return list(rand_set)
#
#     def _augment(self, S, L, n_enum):
#         aug_s, aug_l = [], []
#         for s, lab in zip(S, L):
#             for a in self.enumerate_smiles(s, n_enum):
#                 aug_s.append(a)
#                 aug_l.append(lab)
#         return aug_s, aug_l
#
#     def get_fold_loaders(self, fold_idx, n_enum = 8):
#
#         splits = list(self.kf.split(self.smiles_list))
#         train_idx, test_idx = splits[fold_idx]
#
#         # 获取训练集和测试集
#         smiles_train = [self.smiles_list[i] for i in train_idx]
#         labels_train = [self.label_list[i] for i in train_idx]
#         smiles_test = [self.smiles_list[i] for i in test_idx]
#         labels_test = [self.label_list[i] for i in test_idx]
#
#         # 从训练集中拆分验证集 (10%)
#         smiles_train, smiles_val, labels_train, labels_val = train_test_split(
#             smiles_train, labels_train, test_size=0.1, random_state=42, shuffle=self.shuffle)
#
#         if n_enum > 0:
#             smiles_train, labels_train = self._augment(smiles_train, labels_train, n_enum)
#             smiles_val, labels_val = self._augment(smiles_val, labels_val, n_enum)
#             smiles_test, labels_test = self._augment(smiles_test, labels_test, n_enum)
#
#         # 创建数据集和DataLoader
#         train_dataset = MolDataset(smiles_train, labels_train, contrastive=False)
#         val_dataset = MolDataset(smiles_val, labels_val, contrastive=False)
#         test_dataset = MolDataset(smiles_test, labels_test, contrastive=False)
#
#         train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
#         val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
#         test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
#
#         return train_loader, val_loader, test_loader
#
#
#

