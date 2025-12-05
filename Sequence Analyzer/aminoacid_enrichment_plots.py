# ============================================================
# Amino Acid Enrichment Map between Libraries (6-mer region)
# Author: ChatGPT | November 2025
# ============================================================

from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# ---------------------- USER INPUTS --------------------------
fastq_files = {
    "PreSelection": "/Users/timothyvernon/Downloads/PCGH79_fastq/PCGH79_2_G7_plateHandle.fastq",
    "PostSelection": "/Users/timothyvernon/Downloads/4GSY4B_1_sample_1.fastq"
}
target_motif = "AGAGCTATAGGTCGGCTGAGCTCA"  # search 5'→3'
flank_len = 6                              # 6 bp downstream (2 amino acids)
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

def compute_aa_frequencies(name, fastq_path):
    """Translate 6-mers to AA pairs and compute frequency matrix."""
    flanks = extract_6mers(fastq_path, target_motif, flank_len)
    print(f"✅ {name}: Extracted {len(flanks)} valid 6-mers.")

    # Translate to 2-amino-acid sequences
    aa_list = []
    for s in flanks:
        try:
            aa = str(Seq(s).translate(to_stop=False))
            if len(aa) == 2:
                aa_list.append(aa)
        except Exception:
            continue

    # Count each dipeptide
    counts = Counter(aa_list)
    total = sum(counts.values())
    freq_df = pd.DataFrame(counts.items(), columns=["AApair", "count"])
    freq_df["freq_percent"] = 100 * freq_df["count"] / total

    # Split into positions 1 and 2
    freq_df["AA1"] = freq_df["AApair"].str[0]
    freq_df["AA2"] = freq_df["AApair"].str[1]

    # Pivot into 2D position heatmap
    matrix = freq_df.pivot_table(index="AA1", columns="AA2", values="freq_percent", fill_value=0)

    # Save results
    os.makedirs("results_" + name, exist_ok=True)
    matrix.to_csv(f"results_{name}/{name}_AA_frequency_matrix.csv")
    print(f"💾 Saved amino acid frequency matrix for {name}")

    return matrix

# --- Compute for each library ---
matrices = {}
for name, path in fastq_files.items():
    matrices[name] = compute_aa_frequencies(name, path)

# --- Align both matrices ---
all_AA1 = sorted(set(matrices["PreSelection"].index).union(matrices["PostSelection"].index))
all_AA2 = sorted(set(matrices["PreSelection"].columns).union(matrices["PostSelection"].columns))

for name in matrices:
    matrices[name] = matrices[name].reindex(index=all_AA1, columns=all_AA2, fill_value=0)

# --- Compute enrichment (Δ Post - Pre) ---
diff_matrix = matrices["PostSelection"] - matrices["PreSelection"]

# --- Reorder axes so AA start (position 1) is top-left ---
diff_matrix = diff_matrix.iloc[::-1]  # flip y-axis so top = N-terminal position 1

# --- Plot enrichment heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(
    diff_matrix,
    cmap="coolwarm",
    center=0,
    annot=False,
    cbar_kws={"label": "Δ Frequency (%) (Post - Pre)"}
)
plt.title("Amino Acid Enrichment Map (Position 1 vs Position 2)")
plt.xlabel("Position 2 (C-terminal AA)")
plt.ylabel("Position 1 (N-terminal AA)")
plt.tight_layout()
plt.savefig("AA_enrichment_map.png", dpi=300)
plt.show()

# --- Save combined data ---
diff_matrix.to_csv("AA_enrichment_matrix.csv")
print("🔥 Saved amino acid enrichment heatmap and data → AA_enrichment_map.png, AA_enrichment_matrix.csv")
