# ============================================================
# Detect full and partial Illumina handle matches (P5/P7),
# quantify reads, reconstruct full-length variants (~210 bp),
# and extract amino acids at target codons.
# ============================================================

from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
from collections import Counter

# ---------- USER INPUTS ----------
fastq_path = "/Users/timothyvernon/Downloads/Z2SFTS_1_G7Lib_Enrichment.fastq"

# Illumina adapter sequences
P5_full = "AATGATACGGCGACCACCGAGATCTACAC"
P7_full = "CAAGCAGAAGACGGCATACGAGAT"

# Partial match threshold (minimum bp match)
partial_len = 12

# Expected insert size
min_len, max_len = 190, 230

# Target motif for codon extraction
target_motif = "AGAGCTATAGGTCGGCTGAGCTCA"
flank_len = 6  # downstream 6 bp = 2 codons
# ---------------------------------

# --- Counters ---
reads_with_full_handles = 0
reads_with_partial_handles = 0
reads_with_P5_only = 0
reads_with_P7_only = 0

unique_variants = Counter()
partial_records = []

# --- Helper for partial matching ---
def has_partial(seq, adapter, min_overlap):
    for L in range(len(adapter), min_overlap - 1, -1):
        if adapter[:L] in seq:
            return True
    return False

# --- 1️⃣ Scan FASTQ for full & partial handle patterns ---
for record in SeqIO.parse(fastq_path, "fastq"):
    seq = str(record.seq).upper()
    has_P5 = P5_full in seq
    has_P7 = P7_full in seq
    partial_P5 = has_partial(seq, P5_full, partial_len)
    partial_P7 = has_partial(seq, P7_full, partial_len)

    # --- Full handle reads ---
    if has_P5 and has_P7:
        reads_with_full_handles += 1
        start = seq.find(P5_full) + len(P5_full)
        end = seq.find(P7_full)
        insert = seq[start:end]
        if min_len <= len(insert) <= max_len:
            unique_variants[insert] += 1

    # --- Partial matches ---
    elif (partial_P5 and partial_P7):
        reads_with_partial_handles += 1
        partial_records.append((record.id, "both_partial"))
    elif partial_P5:
        reads_with_partial_handles += 1
        reads_with_P5_only += 1
        partial_records.append((record.id, "partial_P5"))
    elif partial_P7:
        reads_with_partial_handles += 1
        reads_with_P7_only += 1
        partial_records.append((record.id, "partial_P7"))
    elif has_P5:
        reads_with_P5_only += 1
    elif has_P7:
        reads_with_P7_only += 1

# --- 2️⃣ Count results ---
print("===== HANDLE SUMMARY =====")
print(f"Reads with full P5 + P7 handles: {reads_with_full_handles}")
print(f"Reads with partial handles (≥{partial_len} bp overlap): {reads_with_partial_handles}")
print(f"Reads with only P5: {reads_with_P5_only}")
print(f"Reads with only P7: {reads_with_P7_only}")
print(f"Unique full-length variants (~{min_len}-{max_len} bp): {len(unique_variants)}")

# --- 3️⃣ Build DataFrame of variants ---
df = pd.DataFrame(unique_variants.items(), columns=["VariantSequence", "Count"])
df.sort_values("Count", ascending=False, inplace=True)

# --- 4️⃣ Extract and translate target codon window ---
aa_records = []
for seq, count in unique_variants.items():
    idx = seq.find(target_motif)
    if idx != -1 and idx + len(target_motif) + flank_len <= len(seq):
        flank = seq[idx + len(target_motif): idx + len(target_motif) + flank_len]
        try:
            aa = str(Seq(flank).translate())
        except Exception:
            aa = "NA"
    else:
        flank, aa = "NA", "NA"
    aa_records.append((seq, count, flank, aa))

aa_df = pd.DataFrame(aa_records, columns=["VariantSequence", "Count", "Target6bp", "AminoAcids"])

# --- 5️⃣ Save results ---
aa_df.to_csv("FullAmplicon_UniqueVariants_withPartial.csv", index=False)
pd.DataFrame(partial_records, columns=["ReadID", "PartialType"]).to_csv("PartialHandle_Reads.csv", index=False)

# --- 6️⃣ Summary ---
total_reads = sum(1 for _ in SeqIO.parse(fastq_path, "fastq"))
print("\n===== SUMMARY =====")
print(f"Total FASTQ reads: {total_reads}")
print(f"Reads with full handles: {reads_with_full_handles} ({reads_with_full_handles/total_reads:.2%})")
print(f"Reads with partial handles: {reads_with_partial_handles} ({reads_with_partial_handles/total_reads:.2%})")
print(f"Unique full-length (~210 bp) variants: {len(unique_variants)}")
print(f"Variants with valid codon translation: {aa_df[aa_df['AminoAcids'] != 'NA'].shape[0]}")
print("\nExample variants:")
print(aa_df.head(5))
