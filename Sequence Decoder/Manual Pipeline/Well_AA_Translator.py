import os
import pandas as pd
from Bio.Seq import Seq
from Bio.Data import CodonTable
from TargetRegion_FullLibrary import output_csv
from Barcode_Well_Identification import output_csv_final
from TargetRegion_FullLibrary import basename

# === USER INPUT ===
annotated_csv = output_csv_final  # your annotated file
output_csv = f"Well_AA_Translation_{basename}.csv"
primer_seq = "AGAGCTATAGGTCGGCTGAGCTCA"  # upstream primer (5' → 3')
variant_len = 6  # number of bases after the primer to capture
# ===================

# --- Load data ---
print("📥 Loading annotated CSV...")
df = pd.read_csv(annotated_csv)
print(f"✅ Loaded {len(df)} reads.")

# --- Find variant region (6 bp downstream of primer) ---
def extract_variant(seq, primer=primer_seq, length=variant_len):
    seq = seq.upper()
    idx = seq.find(primer)
    if idx == -1:
        return None
    start = idx + len(primer)
    return seq[start:start + length] if len(seq[start:start + length]) == length else None

print("🧬 Extracting variant regions...")
df["VariantDNA"] = df["FullSequence"].apply(extract_variant)

# --- Drop reads without primer match ---
df = df.dropna(subset=["VariantDNA"])
print(f"✅ {len(df)} reads contain the primer and variant region.")

# --- Translate 6bp variant (two codons) to amino acids ---
def translate_variant(dna_seq):
    try:
        seq_obj = Seq(dna_seq)
        return str(seq_obj.translate(table=CodonTable.unambiguous_dna_by_name["Standard"]))
    except Exception:
        return None

df["VariantAA"] = df["VariantDNA"].apply(translate_variant)

# --- Group by Well ---
print("📊 Grouping by Well ID...")
summary = (
    df.groupby(["Well", "VariantAA"])
    .size()
    .reset_index(name="ReadCount")
    .sort_values(["Well", "ReadCount"], ascending=[True, False])
)

# --- Save results ---
summary.to_csv(output_csv, index=False)
print(f"💾 Saved per-well amino acid table → {output_csv}")

# --- Optional: print top variants per well ---
print("\n🔝 Top variants per well:")
for well in summary["Well"].unique()[:8]:  # show first 8 wells for preview
    subset = summary[summary["Well"] == well].head(3)
    print(f"  {well}: {', '.join([f'{aa} ({n})' for aa, n in zip(subset['VariantAA'], subset['ReadCount'])])}")
