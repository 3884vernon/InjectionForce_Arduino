from Bio import SeqIO
import pandas as pd
import numpy as np
import os

# === USER INPUTS ===

# Insert fastQ directory
dir = "/Users/timothyvernon/Downloads"

# Insert filename
filename = "4GSY4B_1_sample_1.fastq"

# Join directory and filename
fastq_file = os.path.join(dir, filename)

# Create basename for variable file naming
basename = os.path.splitext(filename)[0]

# Input the target primer sequence 5' to 3'
target_primer = "AGAGCTATAGGTCGGCTGAGCTCA".upper()
flank_len = 6  # number of nucleotides downstream

# --- CHANGE THIS OUTPUT FILE NAME TO FIT YOUR INPUT LIBRARY ---
output_csv = f"TargetPrimer_Hits_{basename}.csv"

# ====================

results = []

# --- Scan FASTQ ---
for record in SeqIO.parse(fastq_file, "fastq"):
    seq = str(record.seq).upper()
    if target_primer in seq:
        idx = seq.find(target_primer)
        downstream = seq[idx + len(target_primer): idx + len(target_primer) + flank_len]
        full_hit = seq[idx: idx + len(target_primer) + flank_len]
        results.append({
            "ReadID": record.id,
            "FullSequence": seq,
            "PrimerSequence": target_primer,
            "Downstream6bp": downstream,
            "HitRegion": full_hit
        })

# --- Save results ---
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"✅ Found {len(df)} reads containing the primer sequence.")
print(f"💾 Saved → {output_csv}")

# --- Optional summary ---
if len(df) > 0:
    print("\nTop 10 downstream 6bp variants:")
    print(df["Downstream6bp"].value_counts().head(10))

