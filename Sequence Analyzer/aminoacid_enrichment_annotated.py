# ============================================================
# Annotated Amino Acid Enrichment Map (Post - Pre) with Counts
# Author: ChatGPT | November 2025
# ============================================================

from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import numpy as np

# ---------------------- USER INPUTS --------------------------
fastq_files = {
    "PreSelection": "/Users/timothyvernon/Downloads/PCGH79_fastq/PCGH79_2_G7_plateHandle.fastq",
    "PostSelection": "/Users/timothyvernon/Downloads/4GSY4B_1_sample_1.fastq"
}
target_motif = "AGAGCTATAGGTCGGCTGAGCTCA"  # 5'→3'
flank_len = 6                              # 6 bp downstream = 2 codons
# -------------------------------------------------------------

def extract_6mers(fastq_path, motif, flank_len):
    """Extract 6 bp downstream of the motif from FASTQ reads."""
    flanks = []
    for record in SeqIO.parse(fastq_path, "fastq"):
        seq = str(record.seq).upper()
        idx = seq.find(motif)
        if idx != -1 and idx + len(motif) + flank_len <= len(seq):
            flank = seq[idx + len(motif): idx + len(motif) + flank_len]
            if "N" not in flank:
                flanks.append(flank)
    return flanks

def compute_aa_counts(name, fastq_path):
    """Translate 6-mers to AA pairs and compute frequency and count matrix."""
    flanks = extract_6mers(fastq_path, target_motif, flank_len)
    print(f"✅ {name}: Extracted {len(flanks)} valid 6-mers.")

    # Translate to amino acids (2 residues)
    aa_pairs = []
    for s in flanks:
        try:
            aa = str(Seq(s).translate(to_stop=False))
            if len(aa) == 2:
                aa_pairs.append(aa)
        except Exception:
            continue

    # Count all dipeptides
    counts = Counter(aa_pairs)
    total = sum(counts.values())
    df = pd.DataFrame(counts.items(), columns=["AApair", "count"])
    df["freq_percent"] = 100 * df["count"] / total
    df["AA1"] = df["AApair"].str[0]
    df["AA2"] = df["AApair"].str[1]

    # Pivot to 2D matrices
    freq_matrix = df.pivot_table(index="AA1", columns="AA2", values="freq_percent", fill_value=0)
    count_matrix = df.pivot_table(index="AA1", columns="AA2", values="count", fill_value=0)

    os.makedirs(f"results_{name}", exist_ok=True)
    freq_matrix.to_csv(f"results_{name}/{name}_AA_freq.csv")
    count_matrix.to_csv(f"results_{name}/{name}_AA_counts.csv")
    print(f"💾 Saved amino acid frequencies and counts for {name}")
    return freq_matrix, count_matrix

# --- Compute for each library ---
freq = {}
counts = {}
for name, path in fastq_files.items():
    freq[name], counts[name] = compute_aa_counts(name, path)

# --- Align indices and columns ---
all_AA1 = sorted(set(freq["PreSelection"].index).union(freq["PostSelection"].index))
all_AA2 = sorted(set(freq["PreSelection"].columns).union(freq["PostSelection"].columns))

for name in freq:
    freq[name] = freq[name].reindex(index=all_AA1, columns=all_AA2, fill_value=0)
    counts[name] = counts[name].reindex(index=all_AA1, columns=all_AA2, fill_value=0)

# --- Compute enrichment and Δ counts ---
# diff_freq = freq["PostSelection"] - freq["PreSelection"]
diff_counts = counts["PostSelection"] - counts["PreSelection"]

diff_freq = freq["PostSelection"] - freq["PreSelection"]
diff_counts = counts["PostSelection"] - counts["PreSelection"]

# --- Flip Y-axis so N-terminal (Position 1) starts top-left ---
diff_freq = diff_freq.iloc[::-1]
diff_counts = diff_counts.iloc[::-1]

# --- Plot annotated enrichment map ---
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    diff_freq,
    cmap="coolwarm",
    center=0,
    annot=diff_counts.astype(int),
    fmt="d",
    linewidths=0.5,
    cbar_kws={"label": "Δ Frequency (%) (Post - Pre)"}
)
plt.title("Amino Acid Enrichment Map (Position 1 vs Position 2)")
plt.xlabel("Position 2 (C-terminal AA)")
plt.ylabel("Position 1 (N-terminal AA)")
plt.tight_layout()
plt.savefig("AA_enrichment_map_annotated.png", dpi=300)
plt.show()

# --- Save combined data ---
diff_freq.to_csv("AA_enrichment_matrix_percent.csv")
diff_counts.to_csv("AA_enrichment_matrix_counts.csv")

print("🔥 Saved annotated amino acid enrichment map and data → AA_enrichment_map_annotated.png")
