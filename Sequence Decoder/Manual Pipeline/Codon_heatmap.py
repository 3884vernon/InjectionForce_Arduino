# ============================================================
# Make a heatmap of amino acid enrichment at codon 1 and codon 2
# from the downstream 6bp region in TargetPrimer_Hits.csv
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.Seq import Seq
from TargetRegion_FullLibrary import output_csv
from TargetRegion_FullLibrary import basename

# === USER INPUT === Please BE SURE TO CHANGE OR WILL OVERWRITE YOUR FILES
input_csv = output_csv
output_fig = f"AminoAcid_CodonHeatmap_{basename}.png"
# ===================

# --- Load the data ---
df = pd.read_csv(input_csv)

# Drop any missing or short downstream sequences
df = df[df["Downstream6bp"].str.len() == 6]

# --- Translate each 6bp region into amino acids ---
aa_pairs = []
for seq in df["Downstream6bp"]:
    try:
        aa = str(Seq(seq).translate())
        if len(aa) == 2:
            aa_pairs.append(aa)
    except Exception:
        pass

if len(aa_pairs) == 0:
    raise ValueError("No valid 6bp sequences found that translate into 2 amino acids.")

# --- Build frequency table ---
aa1 = [a[0] for a in aa_pairs]
aa2 = [a[1] for a in aa_pairs]
aa_letters = sorted(set(aa1 + aa2))

freq_df = pd.DataFrame(0, index=aa_letters, columns=aa_letters)

# Count co-occurrences
for pair in aa_pairs:
    a1, a2 = pair[0], pair[1]
    freq_df.loc[a1, a2] += 1

# Normalize to percentages
freq_norm = freq_df / freq_df.sum().sum() * 100

# --- Plot heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(
    freq_norm,
    cmap="viridis",
    annot=freq_df,
    fmt=".0f",
    linewidths=0.5,
    cbar_kws={'label': 'Frequency (%)'}
)
plt.title("Amino Acid Enrichment at Codon 1 (Y-axis) vs Codon 2 (X-axis)")
plt.xlabel("Codon 2 Amino Acid")
plt.ylabel("Codon 1 Amino Acid")
plt.tight_layout()
plt.savefig(output_fig, dpi=300)
plt.show()

print(f"✅ Saved amino-acid codon heatmap → {output_fig}")



