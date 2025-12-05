# ============================================================
# Analyze downstream 6-mers from FASTQ and visualize codon & amino acid frequencies
# Author: ChatGPT | November 2025
# ============================================================

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ----------------------- USER INPUTS -------------------------
fastq_path = "/Users/timothyvernon/Downloads/Z2SFTS_1_G7Lib_Enrichment.fastq"
target_motif = "AGAGCTATAGGTCGGCTGAGCTCA"   # search in 5'→3'
flank_len = 6                               # extract 6 bp downstream
# -------------------------------------------------------------

# 1️⃣  Extract downstream 6-mer sequences
flanks = []
for record in SeqIO.parse(fastq_path, "fastq"):
    seq = str(record.seq).upper()
    idx = seq.find(target_motif)
    if idx != -1 and idx + len(target_motif) + flank_len <= len(seq):
        flank = seq[idx + len(target_motif): idx + len(target_motif) + flank_len]
        if "N" not in flank:
            flanks.append(flank)

print(f"✅ Found {len(flanks)} total hits with valid 6-mers.")

# 2️⃣  Count occurrences
counts = Counter(flanks)
df = pd.DataFrame(counts.items(), columns=["6mer", "count"]).sort_values("count", ascending=False)
df.reset_index(drop=True, inplace=True)

# 3️⃣  Translate 6-mers to amino acids (2 residues)
aa_list = []
codon_records = []  # track codon → amino acid mapping
for s in df["6mer"]:
    try:
        aa = str(Seq(s).translate(to_stop=False))
        if len(aa) == 2:  # valid translation
            count = int(df.loc[df["6mer"] == s, "count"].values[0])
            aa_list.extend([aa] * count)
            codon_records.extend([(s[:3], aa[0], count), (s[3:], aa[1], count)])
    except Exception:
        continue

print(f"🧩 Translated {len(aa_list)} amino acid pairs.")

# -----------------------------------------------------------
# 🧮 AMINO ACID FREQUENCY HEATMAP
# -----------------------------------------------------------

aa_positions = {1: [], 2: []}
for seq in aa_list:
    aa_positions[1].append(seq[0])
    aa_positions[2].append(seq[1])

aa_letters = sorted(set("".join(aa_list)))
freq_matrix = pd.DataFrame(0, index=aa_letters, columns=["Pos1", "Pos2"])

for pos in [1, 2]:
    col = f"Pos{pos}"
    counts = pd.Series(aa_positions[pos]).value_counts()
    for aa, c in counts.items():
        freq_matrix.loc[aa, col] = c

freq_matrix = freq_matrix.div(freq_matrix.sum(axis=0), axis=1) * 100

# Physicochemical classes
aa_classes = {
    "S": "Polar", "T": "Polar", "N": "Polar", "Q": "Polar", "C": "Polar", "Y": "Polar",
    "A": "Hydrophobic", "V": "Hydrophobic", "L": "Hydrophobic", "I": "Hydrophobic",
    "M": "Hydrophobic", "F": "Hydrophobic", "W": "Hydrophobic", "P": "Hydrophobic", "G": "Hydrophobic",
    "K": "Charged", "R": "Charged", "H": "Charged", "D": "Charged", "E": "Charged"
}
freq_matrix["Class"] = freq_matrix.index.map(lambda x: aa_classes.get(x, "Other"))
order = ["Charged", "Polar", "Hydrophobic", "Other"]
freq_matrix["Class"] = pd.Categorical(freq_matrix["Class"], categories=order, ordered=True)
freq_matrix.sort_values("Class", inplace=True)

# --- Plot AA heatmap ---
plt.figure(figsize=(6, 6))
sns.heatmap(freq_matrix[["Pos1", "Pos2"]], cmap="viridis", annot=True, fmt=".1f",
            cbar_kws={"label": "Frequency (%)"})
plt.title("Amino Acid Frequency per Position (6-mer downstream region)")
plt.xlabel("Position in Translated Peptide")
plt.ylabel("Amino Acid")
plt.tight_layout()
plt.savefig("6mer_aminoacid_heatmap.png", dpi=300)
plt.show()

# --- Class sidebar ---
plt.figure(figsize=(1.5, 6))
sns.heatmap(
    pd.DataFrame(freq_matrix["Class"].map({
        "Charged": 1, "Polar": 2, "Hydrophobic": 3, "Other": 4
    })).T,
    cmap=sns.color_palette(["#5E4FA2", "#66C2A5", "#FC8D62", "#E78AC3"], as_cmap=True),
    cbar=False,
    xticklabels=freq_matrix.index,
    yticklabels=["Class"]
)
plt.title("Residue Classes")
plt.tight_layout()
plt.savefig("6mer_class_sidebar.png", dpi=300)
plt.show()

# -----------------------------------------------------------
# 🧬 CODON-LEVEL HEATMAP
# -----------------------------------------------------------

# Expand codon records into DataFrame
codon_df = pd.DataFrame(codon_records, columns=["Codon", "AminoAcid", "count"])
codon_df = codon_df.groupby(["Codon", "AminoAcid"], as_index=False)["count"].sum()

# Get codon table for context
std_table = CodonTable.unambiguous_dna_by_name["Standard"].forward_table

# Ensure only valid codons
codon_df = codon_df[codon_df["Codon"].isin(std_table.keys())]

# Normalize counts to percent
codon_df["Percent"] = 100 * codon_df["count"] / codon_df["count"].sum()

# Pivot to make Codon × AminoAcid matrix
codon_matrix = codon_df.pivot_table(index="Codon", columns="AminoAcid", values="Percent", fill_value=0)

# Sort codons by encoded amino acid
codon_matrix = codon_matrix.loc[sorted(codon_matrix.index, key=lambda x: std_table.get(x, "Z"))]

# Plot codon heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(codon_matrix, cmap="mako", cbar_kws={"label": "Frequency (%)"})
plt.title("Codon Usage Frequency for Downstream 6-mer Region")
plt.xlabel("Amino Acid")
plt.ylabel("Codon")
plt.tight_layout()
plt.savefig("6mer_codon_heatmap.png", dpi=300)
plt.show()

# -----------------------------------------------------------
# SAVE ALL RESULTS
# -----------------------------------------------------------
freq_matrix.to_csv("6mer_aminoacid_frequencies.csv")
codon_matrix.to_csv("6mer_codon_frequencies.csv")
print("💾 Saved amino acid and codon frequency data.")
print("🔥 Figures saved → amino acid heatmap, class sidebar, and codon heatmap.")
