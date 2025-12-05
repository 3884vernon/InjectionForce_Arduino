# ============================================================
# Compare codon & amino-acid usage across multiple FASTQ libraries
# Author: ChatGPT | November 2025
# ============================================================

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# ---------------------- USER INPUTS --------------------------
fastq_files = {
    "PreSelection": "/Users/timothyvernon/Downloads/PCGH79_fastq/PCGH79_2_G7_plateHandle.fastq",
    "PostSelection": "/Users/timothyvernon/Downloads/Z2SFTS_1_G7Lib_Enrichment.fastq"
}
target_motif = "AGAGCTATAGGTCGGCTGAGCTCA"
flank_len = 6


# -------------------------------------------------------------

def extract_6mers(fastq_path, motif, flank_len):
    flanks = []
    for record in SeqIO.parse(fastq_path, "fastq"):
        seq = str(record.seq).upper()
        idx = seq.find(motif)
        if idx != -1 and idx + len(motif) + flank_len <= len(seq):
            flank = seq[idx + len(motif): idx + len(motif) + flank_len]
            if "N" not in flank:
                flanks.append(flank)
    return flanks


def analyze_library(name, fastq_path):
    print(f"\n=== Processing {name} ===")
    flanks = extract_6mers(fastq_path, target_motif, flank_len)
    print(f"Found {len(flanks)} valid 6-mers in {name}")
    counts = Counter(flanks)
    df = pd.DataFrame(counts.items(), columns=["6mer", "count"])
    df.sort_values("count", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Translate 6-mers into AA pairs and codons
    aa_list = []
    codon_records = []
    for s in df["6mer"]:
        try:
            aa = str(Seq(s).translate(to_stop=False))
            if len(aa) == 2:
                count = int(df.loc[df["6mer"] == s, "count"].values[0])
                aa_list.extend([aa] * count)
                codon_records.extend([(s[:3], aa[0], count), (s[3:], aa[1], count)])
        except Exception:
            continue

    codon_df = pd.DataFrame(codon_records, columns=["Codon", "AminoAcid", "count"])
    codon_df = codon_df.groupby(["Codon", "AminoAcid"], as_index=False)["count"].sum()

    std_table = CodonTable.unambiguous_dna_by_name["Standard"].forward_table
    codon_df = codon_df[codon_df["Codon"].isin(std_table.keys())]
    codon_df["Percent"] = 100 * codon_df["count"] / codon_df["count"].sum()

    codon_matrix = codon_df.pivot_table(index="Codon", columns="AminoAcid",
                                        values="Percent", fill_value=0)

    outdir = "results_" + name
    os.makedirs(outdir, exist_ok=True)
    codon_matrix.to_csv(os.path.join(outdir, f"{name}_codon_frequencies.csv"))
    print(f"Saved {name} codon frequencies.")

    return codon_matrix


# --- Analyze each library ---
codon_matrices = {}
for name, path in fastq_files.items():
    codon_matrices[name] = analyze_library(name, path)

# --- Merge for comparison ---
common_codons = set().union(*[m.index for m in codon_matrices.values()])
common_aas = set().union(*[m.columns for m in codon_matrices.values()])

for name in codon_matrices:
    codon_matrices[name] = codon_matrices[name].reindex(index=sorted(common_codons),
                                                        columns=sorted(common_aas),
                                                        fill_value=0)

# --- Differential codon usage (Post - Pre) ---
diff_matrix = codon_matrices["PostSelection"] - codon_matrices["PreSelection"]

# --- Plot comparative heatmaps ---
plt.figure(figsize=(14, 6))
sns.heatmap(codon_matrices["PreSelection"], cmap="mako",
            cbar_kws={'label': 'Frequency (%)'})
plt.title("Codon Usage - PreSelection Library")
plt.xlabel("Amino Acid");
plt.ylabel("Codon")
plt.tight_layout();
plt.savefig("codon_usage_pre.png", dpi=300);
plt.close()

plt.figure(figsize=(14, 6))
sns.heatmap(codon_matrices["PostSelection"], cmap="mako",
            cbar_kws={'label': 'Frequency (%)'})
plt.title("Codon Usage - PostSelection Library")
plt.xlabel("Amino Acid");
plt.ylabel("Codon")
plt.tight_layout();
plt.savefig("codon_usage_post.png", dpi=300);
plt.close()

plt.figure(figsize=(14, 6))
sns.heatmap(diff_matrix, cmap="coolwarm", center=0,
            cbar_kws={'label': 'Δ Frequency (%) (Post - Pre)'})
plt.title("Differential Codon Usage (Post - Pre)")
plt.xlabel("Amino Acid");
plt.ylabel("Codon")
plt.tight_layout();
plt.savefig("codon_usage_diff.png", dpi=300)
plt.show()

print("🔥 Comparative codon usage plots saved: pre, post, and Δ-difference heatmaps.")
