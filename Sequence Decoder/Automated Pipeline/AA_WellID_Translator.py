import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

# === Universal Codon Table ===
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# === Translation helpers ===
def translate_codon(codon):
    codon = codon.upper().replace("U", "T")
    return CODON_TABLE.get(codon, "X")

def translate_region(seq, primer="AGAGCTATAGGTCGGCTGAGCTCA", downstream_len=6):
    """Extract 6 bp downstream of primer and translate to two amino acids."""
    seq = seq.upper()
    if primer not in seq:
        return "NN"
    start = seq.find(primer) + len(primer)
    region = seq[start:start + downstream_len]
    if len(region) < 6 or not re.match("^[ATGC]+$", region):
        return "NN"
    aa1, aa2 = translate_codon(region[:3]), translate_codon(region[3:])
    return aa1 + aa2


# === Plotting helpers ===
def plot_codon_pair_heatmap(df, basename):
    """Absolute frequency heatmap of AA1 × AA2 codon pairs."""
    aa_pairs = df["AminoAcid"].value_counts().reset_index()
    aa_pairs.columns = ["AA_Pair", "Count"]
    aa_pairs["AA1"] = aa_pairs["AA_Pair"].str[0]
    aa_pairs["AA2"] = aa_pairs["AA_Pair"].str[1]

    pivot = aa_pairs.pivot(index="AA1", columns="AA2", values="Count").fillna(0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".0f",
                cbar_kws={"label": "Absolute Count"})
    plt.title("Amino Acid Pair Frequency (AA1 × AA2)")
    plt.xlabel("Second Codon AA")
    plt.ylabel("First Codon AA")
    plt.tight_layout()
    plot_path = f"AA_Pair_Heatmap_{basename}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🎨 Amino-acid pair heatmap saved: {plot_path}")


def plot_well_variant_heatmap(df, basename):
    """Heatmap of most frequent AA variant per well (absolute read counts)."""
    summary = (
        df.groupby("Well")["AminoAcid"]
        .apply(lambda x: x.value_counts().idxmax())
        .reset_index()
    )
    counts = df.groupby("Well")["AminoAcid"].count().reset_index()
    summary = summary.merge(counts, on="Well", suffixes=("", "_Count"))
    summary.columns = ["Well", "AminoAcid", "Count"]

    # Extract plate coordinates
    summary = summary[summary["Well"].str.match(r"^[A-Ha-h]\d{1,2}$")]
    summary["Row"] = summary["Well"].str[0].str.upper()
    summary["Col"] = summary["Well"].str[1:].astype(int)

    plate = summary.pivot(index="Row", columns="Col", values="Count").reindex(
        index=list("ABCDEFGH"), columns=range(1, 13)
    ).fillna(0)

    plt.figure(figsize=(15, 6))
    sns.heatmap(plate, annot=True, fmt=".0f", cmap="viridis",
                cbar_kws={"label": "Read Count"})
    plt.title("Dominant Variant Frequency by Well")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plot_path = f"AA_Well_Heatmap_{basename}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🧫 Well-variant heatmap saved: {plot_path}")


# === Main pipeline ===
def main(input_csv=None, basename="output"):
    if input_csv is None:
        raise ValueError("❌ Please provide input_csv to Well_AA_Translator.main()")

    print(f"📥 Loading annotated CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    df["AminoAcid"] = df["FullSequence"].apply(translate_region)

    # Save summarized data
    aa_summary = (
        df.groupby("Well")["AminoAcid"]
        .apply(lambda x: x.value_counts().head(3))
        .reset_index()
    )
    aa_summary.columns = ["Well", "AminoAcid", "Count"]
    summary_csv = f"AA_Translation_Summary_{basename}.csv"
    aa_summary.to_csv(summary_csv, index=False)
    print(f"✅ Amino-acid summary saved: {summary_csv}")

    # Generate plots
    plot_codon_pair_heatmap(df, basename)
    plot_well_variant_heatmap(df, basename)

    print("✅ Amino-acid translation and visualization complete.")
    return summary_csv


if __name__ == "__main__":
    main()
