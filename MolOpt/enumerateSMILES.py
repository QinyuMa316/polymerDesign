"""
[
'*C(C)(C(=O)OC)C*',
'*CC(C)(*)C(OC)=O',
'*CC(C)(*)C(=O)OC',
'C(*)C(C(OC)=O)(*)C',
'*CC(*)(C)C(=O)OC',
'*C(C*)(C)C(=O)OC',
'*C(C*)(C(OC)=O)C',
'COC(C(C*)(C)*)=O']
"""
from rdkit import Chem
def enumerate_smiles(smi: str, n: int = 8):
    """
    给定 canonical SMILES，返回 n 条随机 SMILES（含自身），去重
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []  # 无法解析的直接跳过
    rand_set = {Chem.MolToSmiles(mol, doRandom=True) for _ in range(n)}
    rand_set.add(Chem.MolToSmiles(mol, canonical=True))  # 保障含原始
    return list(rand_set)


if __name__ == '__main__':
    # smi = 'COC(=O)C(C)(*)C*'
    smi = 'COC(=O)C(C)(*)C*'
    print(enumerate_smiles(smi, n=8))


