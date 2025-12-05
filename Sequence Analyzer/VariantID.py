from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import plotly.express as px
import math

# ========== USER SETTINGS ==========
fastq_path = "/Users/timothyvernon/Downloads/PCGH79_fastq/PCGH79_1_G6_plateHandle.fastq"
target_motif = "AGAGCTATAGGTCGGCTGAGCTCA"   # target sequence (5' → 3')
flank_len = 6                               # extract 6 bp downstream
topN = 30                                   # number of top variants to display
# ====================================

# === 1️⃣ Extract 6-bp downstream sequences ===
flanks = []
for record in SeqIO.parse(fastq_path, "fastq"):
    seq = str(record.seq).upper()
    idx = seq.find(target_motif)
    if idx != -1 and idx + len(target_motif) + flank_len <= len(seq):
        flank = seq[idx + len(target_motif): idx + len(target_motif) + flank_len]
        if "N" not in flank:
            flanks.append(flank)

print(f"✅ Found {len(flanks)} total hits with valid downstream 6-mers")

# === 2️⃣ Count occurrences ===
counts = Counter(flanks)
df = pd.DataFrame(counts.items(), columns=["6mer", "count"])
df.sort_values("count", ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

# === 3️⃣ Compute percent abundance ===
total = df["count"].sum()
df["percent"] = 100 * df["count"] / total

# === 4️⃣ Compute Shannon diversity ===
probs = df["count"] / total
shannon_H = -sum(p * math.log(p, 2) for p in probs)
print(f"🧮 Shannon Diversity Index (H): {shannon_H:.3f} bits")

# === 5️⃣ Save results ===
df.to_csv("6mer_counts_enhanced.csv", index=False)
print("💾 Saved data → 6mer_counts_enhanced.csv")

# === 6️⃣ Ranked log plot ===
plt.figure(figsize=(10,5))
plt.plot(range(len(df)), df["count"], lw=1.2)
plt.yscale("log")
plt.xlabel("6-mer Rank")
plt.ylabel("Occurrence (log scale)")
plt.title("Distribution of 6-bp sequences downstream of motif")
plt.tight_layout()
plt.savefig("6mer_rank_distribution.png", dpi=300)
plt.show()

# === 7️⃣ Top-N bar chart ===
plt.figure(figsize=(12,5))
sns.barplot(data=df.head(topN), x="6mer", y="count", palette="viridis")
plt.xticks(rotation=90)
plt.ylabel("Count")
plt.title(f"Top {topN} 6-bp sequences downstream of motif")
plt.tight_layout()
plt.savefig("6mer_topN_bar.png", dpi=300)
plt.show()

# === 8️⃣ Interactive scatter chart (Plotly) ===
fig = px.scatter(
    df,
    x=df.index,
    y="count",
    hover_data=["6mer", "percent"],
    title="Interactive distribution of downstream 6-mers",
    labels={"x": "Rank", "count": "Occurrences"},
    log_y=True,
    color="percent",
    color_continuous_scale="Viridis",
)
fig.write_html("6mer_interactive_plot.html")
print("🌐 Saved interactive chart → 6mer_interactive_plot.html")

# === 9️⃣ Summary output ===
print("\n===== SUMMARY =====")
print(f"Unique 6-mers: {len(df)}")
print(f"Total motif hits: {total}")
print(f"Shannon Diversity (H): {shannon_H:.3f} bits")
print(f"Top 5 sequences:\n{df.head(5)}")
