import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance
from multiprocessing import Pool, cpu_count, freeze_support
from tqdm import tqdm

def barcode_in_sequence(seq, barcode, max_mismatch=1):
    b_len = len(barcode)
    seq = seq.upper()
    for i in range(0, len(seq) - b_len + 1):
        window = seq[i:i + b_len]
        if levenshtein_distance(window, barcode) <= max_mismatch:
            return True
    return False


def find_barcode_pair_fuzzy(args):
    seq, barcodes, max_mismatch = args
    for _, row in barcodes.iterrows():
        p5 = row["P5_Barcode"]
        p7 = row["P7_Barcode"]
        if barcode_in_sequence(seq, p5, max_mismatch) and barcode_in_sequence(seq, p7, max_mismatch):
            return row["Well"]
    return "Unassigned"


def parallel_barcode_assignment(sequences, barcodes, max_mismatch, num_processes=None):
    if num_processes is None or num_processes <= 0:
        num_processes = max(1, cpu_count() - 1)
    print(f"⚙️ Using {num_processes} CPU cores...")
    tasks = [(seq, barcodes, max_mismatch) for seq in sequences]
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(find_barcode_pair_fuzzy, tasks),
                            total=len(sequences),
                            desc="🔍 Assigning barcodes",
                            dynamic_ncols=True))
    return results


def generate_plate_heatmap(summary_df, output_file):
    valid_rows = summary_df["Well"].astype(str)
    valid_mask = valid_rows.str.match(r"^[A-Ha-h][0-9]{1,2}$")
    valid_df = summary_df[valid_mask].copy()
    if valid_df.empty:
        print("⚠️ No valid wells found; skipping heatmap.")
        return

    wells = pd.DataFrame({
        "Row": [w[0].upper() for w in valid_df["Well"]],
        "Col": [int(w[1:]) for w in valid_df["Well"]],
        "ReadCount": valid_df["ReadCount"].tolist()
    })

    plate = wells.pivot(index="Row", columns="Col", values="ReadCount").reindex(
        index=list("ABCDEFGH"), columns=range(1, 13)
    ).fillna(0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(plate, annot=True, fmt=".0f", cmap="viridis",
                cbar_kws={'label': 'Read Count'})
    plt.title("📊 96-Well Plate Read Count Heatmap")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"🧫 Heatmap saved: {output_file}")


def main(input_csv=None, barcode_file=None, sheet_name="xGen UDI Primers Plate 1",
         max_mismatches=1, threads=0, basename="output"):
    """Assign reads to wells using barcode file and generate heatmap."""
    if input_csv is None or barcode_file is None:
        raise ValueError("❌ Missing required arguments for Barcode_Well_Identification.main()")

    output_csv_final = f"TargetPrimer_Hits_Annotated_{basename}.csv"
    summary_csv = f"Well_ReadCount_Summary_{basename}.csv"
    heatmap_png = f"Well_ReadCount_Heatmap_{basename}.png"

    print(f"📥 Loading sequences from {input_csv}")
    df = pd.read_csv(input_csv)
    df["FullSequence"] = df["FullSequence"].astype(str).str.upper()

    print(f"📖 Loading barcode sheet '{sheet_name}' from {barcode_file}...")
    barcodes = pd.read_excel(barcode_file, sheet_name=sheet_name, skiprows=3, engine="openpyxl")
    barcodes = barcodes.rename(columns={
        "Well position": "Well",
        "i5 index\nForward *": "P5_Barcode",
        "i7 index": "P7_Barcode"
    })[["Well", "P5_Barcode", "P7_Barcode"]].dropna()

    print(f"✅ Loaded {len(barcodes)} barcode pairs.")

    df["Well"] = parallel_barcode_assignment(df["FullSequence"].tolist(), barcodes, max_mismatches, threads)
    df.to_csv(output_csv_final, index=False)
    print(f"✅ Annotated CSV saved: {output_csv_final}")

    summary = df["Well"].value_counts().reset_index()
    summary.columns = ["Well", "ReadCount"]
    summary.to_csv(summary_csv, index=False)
    print(f"📄 Summary saved: {summary_csv}")

    generate_plate_heatmap(summary, heatmap_png)

    print("✅ Barcode assignment complete.")
    return output_csv_final  # <-- return for next stage


if __name__ == "__main__":
    freeze_support()
    main()
