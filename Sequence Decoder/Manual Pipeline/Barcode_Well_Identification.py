import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from Levenshtein import distance as levenshtein_distance
from multiprocessing import Pool, cpu_count, freeze_support
from tqdm import tqdm
from TargetRegion_FullLibrary import basename, output_csv  # imported from your previous script

output_csv_final = f"TargetPrimer_Hits_Annotated_{basename}.csv"

# --- Helper functions ---

def barcode_in_sequence(seq, barcode, max_mismatch=1):
    """Check if a barcode is present in the sequence allowing mismatches."""
    b_len = len(barcode)
    seq = seq.upper()
    for i in range(0, len(seq) - b_len + 1):
        window = seq[i:i + b_len]
        if levenshtein_distance(window, barcode) <= max_mismatch:
            return True
    return False


def find_barcode_pair_fuzzy(args):
    """Find which well corresponds to this sequence based on P5/P7 match."""
    seq, barcodes, max_mismatch = args
    for _, row in barcodes.iterrows():
        p5 = row["P5_Barcode"]
        p7 = row["P7_Barcode"]
        if barcode_in_sequence(seq, p5, max_mismatch) and barcode_in_sequence(seq, p7, max_mismatch):
            return row["Well"]
    return "Unassigned"


def parallel_barcode_assignment(sequences, barcodes, max_mismatch, num_processes=None):
    """Run barcode assignment in parallel using multiple cores."""
    if num_processes is None or num_processes <= 0:
        num_processes = max(1, cpu_count() - 1)
    print(f"⚙️ Using {num_processes} CPU cores...")

    # Prepare (seq, barcodes, mismatch) tasks
    tasks = [(seq, barcodes, max_mismatch) for seq in sequences]

    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(find_barcode_pair_fuzzy, tasks),
            total=len(sequences),
            desc="🔍 Assigning barcodes",
            dynamic_ncols=True
        ))
    return results


def generate_plate_heatmap(summary_df, output_file):
    """Generate a 96-well plate heatmap (log10 scale)."""

    # Filter out invalid well names (anything not A1–H12)
    valid_rows = summary_df["Well"].astype(str)
    valid_mask = valid_rows.str.match(r"^[A-Ha-h][0-9]{1,2}$")
    valid_df = summary_df[valid_mask].copy()

    if valid_df.empty:
        print("⚠️ No valid well-format entries (A1–H12) found — skipping heatmap.")
        return

    wells = pd.DataFrame({
        "Row": [w[0].upper() for w in valid_df["Well"]],
        "Col": [int(w[1:]) for w in valid_df["Well"]],
        "ReadCount": valid_df["ReadCount"].tolist()
    })

    # Create 8×12 matrix and fill missing wells with 0
    plate = wells.pivot(index="Row", columns="Col", values="ReadCount").reindex(
        index=list("ABCDEFGH"), columns=range(1, 13)
    ).fillna(0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(np.log10(plate + 1), annot=True, fmt=".1f", cmap="viridis",
                cbar_kws={'label': 'log₁₀(Read Count)'})
    plt.title("📊 96-Well Plate Read Count Heatmap (log scale)")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

# --- Main pipeline ---
def main():
    # === USER INPUT ===
    dir = "/Users/timothyvernon/Downloads"
    filename = "idt-master-index-list.xlsx"
    barcode_file = os.path.join(dir, filename)

    sequences_csv = output_csv  # from TargetRegion_FullLibrary
    sheet_name = "xGen UDI Primers Plate 1"

    summary_csv = f"Well_ReadCount_Summary_{basename}.csv"
    heatmap_png = f"Well_ReadCount_Heatmap_{basename}.png"

    MAX_MISMATCHES = 1  # Allowed mismatches per barcode
    THREADS = 0         # Auto-detect cores
    # ===================

    # --- Load sequences ---
    print("📥 Loading sequence hits...")
    df = pd.read_csv(sequences_csv)
    df["FullSequence"] = df["FullSequence"].astype(str).str.upper()

    # --- Load barcodes ---
    print(f"📖 Loading barcode sheet '{sheet_name}' from {barcode_file}...")
    barcodes = pd.read_excel(barcode_file, sheet_name=sheet_name, skiprows=3, engine="openpyxl")

    barcodes = barcodes.rename(columns={
        "Well position": "Well",
        "i5 index\nForward *": "P5_Barcode",
        "i7 index": "P7_Barcode"
    })[["Well", "P5_Barcode", "P7_Barcode"]].dropna()

    barcodes["P5_Barcode"] = barcodes["P5_Barcode"].astype(str).str.upper().str.strip()
    barcodes["P7_Barcode"] = barcodes["P7_Barcode"].astype(str).str.upper().str.strip()

    print(f"✅ Loaded {len(barcodes)} barcode pairs from Excel.")
    print(barcodes.head())

    # --- Assign barcodes ---
    print("🚀 Starting fuzzy barcode matching...")
    df["Well"] = parallel_barcode_assignment(df["FullSequence"].tolist(), barcodes, MAX_MISMATCHES, THREADS)

    # --- Save annotated file ---
    df.to_csv(output_csv_final, index=False)
    print(f"\n✅ Annotated file saved: {output_csv_final}")

    # --- Summarize read counts ---
    summary = df["Well"].value_counts().reset_index()
    summary.columns = ["Well", "ReadCount"]
    summary.to_csv(summary_csv, index=False)
    print(f"📄 Well summary saved: {summary_csv}")

    # --- Compute stats ---
    total = len(df)
    assigned = (df["Well"] != "Unassigned").sum()
    percent = 100 * assigned / total
    print(f"\n📈 Assignment rate: {percent:.2f}% ({assigned}/{total} reads)")

    # --- Generate 96-well plate heatmap ---
    generate_plate_heatmap(summary, heatmap_png)

    print("\n✅ Pipeline complete.")
    print(f"  • Annotated CSV: {output_csv_final}")
    print(f"  • Summary CSV:   {summary_csv}")
    print(f"  • Heatmap PNG:   {heatmap_png}")


# --- Required for macOS multiprocessing ---
if __name__ == "__main__":
    freeze_support()
    main()

print(output_csv_final)