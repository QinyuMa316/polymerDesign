
from MolOpt import sascorer
from rdkit import Chem

def calSAScore(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    score = sascorer.calculateScore(mol)
    return score



if __name__ == '__main__':
    # mol_id0 = "*c1ccc(Oc2ccc(-n3c(=O)c4cc5c(=O)n(*)c(=O)c5cc4c3=O)cc2)cc1"
    # SAScore: 3.29
    mol_id0_opt = '*c1cc(C#C)c(Oc2cc(C#C)c(-n3c(=O)c4c(C#C)c5c(=O)n(*)c(=O)c5c(C#C)c4c3=O)c(C#C)c2C#C)c(N)c1N'
    # SAScore: 5.00
    smiles = mol_id0_opt
    score = calSAScore(smiles)
    print(f"SAScore: {score:.2f}")

