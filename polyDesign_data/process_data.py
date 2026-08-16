import pandas as pd
from rdkit import Chem


def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None


def process_file(input_file, output_file):
    df = pd.read_csv(input_file)
    print(f"Original number of records: {len(df)}")

    df["SMILES"] = df["SMILES"].apply(canonicalize)

    invalid = df["SMILES"].isna().sum()
    if invalid:
        print(f"Number of invalid SMILES: {invalid}")

    df = (
        df.dropna(subset=["SMILES"])
        .drop_duplicates(subset="SMILES", keep="first")
    )

    print(f"Number of records after canonicalization and deduplication: {len(df)}")

    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    for target in ["Tg", "EPS"]:
        print(f"Target: {target}")
        process_file(
            f"{target}_combined.csv",
            f"{target}_combined_unique.csv"
        )

    print("Target: PI1M")
    process_file(
        "PI1M.csv",
        "PI1M_unique.csv"
    )

    # Target: PI1M
    # Original number of records: 995799
    # Number of invalid SMILES: 7024
    # Number of records after canonicalization and deduplication: 988775