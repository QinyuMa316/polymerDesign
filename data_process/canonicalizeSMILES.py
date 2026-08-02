import os

import pandas as pd
from rdkit import Chem

def canonicalize_smiles(smiles):
    """Convert a SMILES string to canonical SMILES."""
    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )

def main():

    # input_file = args.input
    # output_file = args.output

    input_file = 'data_modification/polyimides_pred_prop_base_str.csv'
    output_file = 'data_modification/polyimides_pred_prop_base_str_std_test.csv'

    if not os.path.isfile(input_file):
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    df = pd.read_csv(input_file)

    if "SMILES" not in df.columns:
        raise ValueError(
            "The input CSV does not contain a 'SMILES' column. "
            f"Available columns: {list(df.columns)}"
        )

    # # Save the original SMILES column.
    # df.insert(
    #     df.columns.get_loc("SMILES") + 1,
    #     "SMILES_original",
    #     df["SMILES"]
    # )

    # Convert SMILES to canonical SMILES.
    df["SMILES"] = df["SMILES"].apply(canonicalize_smiles)

    invalid_mask = df["SMILES"].isna()
    invalid_count = int(invalid_mask.sum())

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_file, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Successfully canonicalized: {len(df) - invalid_count}")
    print(f"Failed to parse: {invalid_count}")
    print(f"Output saved to: {output_file}")

    if invalid_count > 0:
        output_name, output_ext = os.path.splitext(output_file)
        invalid_file = output_name + "_invalid" + output_ext

        df.loc[invalid_mask].to_csv(
            invalid_file,
            index=False
        )

        print(f"Invalid rows saved to: {invalid_file}")


if __name__ == "__main__":
    main()


