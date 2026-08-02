import csv
import os


def read_csv(file_path):
    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def format_tg(value):
    """Convert values such as 661.0 to 661 while retaining 761.5."""
    value = value.strip()

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

        return str(number)
    except ValueError:
        return value


def combine_data(target):

    # Original Tg dataset containing the columns SMILES and Tg/K.
    raw_data_file = f"raw/{target}.csv"

    # Dataset containing Polymer, EPS, Tg/C, Tg/K, and SMILES.
    source_file = "paper/polyimides_dielectric_tg_final.csv"

    # Merged output file.
    os.makedirs('train_data', exist_ok=True)
    output_file = f"train_data/{target}_combined.csv"

    if not os.path.isfile(raw_data_file):
        raise FileNotFoundError(f"File not found: {raw_data_file}")

    if not os.path.isfile(source_file):
        raise FileNotFoundError(f"File not found: {source_file}")

    raw_rows = read_csv(raw_data_file)
    source_rows = read_csv(source_file)

    new_rows = []

    for row in source_rows:
        smiles = (row.get("SMILES") or "").strip()

        if target == 'Tg':
            target_col_name = 'Tg/K'
        # if target == 'EPS':
        else:
            target_col_name = target

        target_col = (row.get(target_col_name) or "").strip()

        # Skip rows with missing SMILES or Tg/K values.
        if not smiles or not target_col:
            continue

        new_rows.append({
            "SMILES": smiles,
            target_col_name: format_tg(target_col)
        })

    # Append the extracted rows to the end of Tg.csv.
    merged_rows = raw_rows + new_rows

    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["SMILES", target_col_name]
        )

        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Original rows in {target}.csv: {len(raw_rows)}")
    print(f"Rows appended: {len(new_rows)}")
    print(f"Total rows: {len(merged_rows)}")
    print(f"Merged file saved to: {output_file}")


if __name__ == "__main__":
    combine_data(target='EPS')
    combine_data(target='Tg')
    # Original rows in EPS.csv: 382
    # Rows appended: 6
    # Total rows: 388
    # Merged file saved to: train_data/EPS_combined.csv
    # Original rows in Tg.csv: 269
    # Rows appended: 6
    # Total rows: 275
    # Merged file saved to: train_data/Tg_combined.csv