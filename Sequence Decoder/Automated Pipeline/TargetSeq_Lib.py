import os
import pandas as pd
from tqdm import tqdm

# === Global variables exported for pipeline ===
basename = None
output_csv = None

def main(fastq_path=None):
    """Extract sequences with target primer region from FASTQ file."""
    global basename, output_csv

    if fastq_path is None:
        raise ValueError("❌ Please provide a FASTQ file path to TargetRegion_FullLibrary.main()")

    basename = os.path.splitext(os.path.basename(fastq_path))[0]
    output_csv = f"TargetPrimer_Hits_{basename}.csv"
    primer = "AGAGCTATAGGTCGGCTGAGCTCA"

    print(f"📂 Reading FASTQ: {fastq_path}")
    print(f"🔍 Searching for primer: {primer}")

    sequences = []
    with open(fastq_path, "r") as f:
        for line in tqdm(f, desc="Scanning FASTQ"):
            line = line.strip().upper()
            if primer in line:
                idx = line.find(primer)
                downstream = line[idx + len(primer): idx + len(primer) + 6]
                sequences.append({"FullSequence": line, "Downstream6bp": downstream})

    df = pd.DataFrame(sequences)
    df.to_csv(output_csv, index=False)

    print(f"✅ Found {len(df)} reads containing the primer sequence.")
    print(f"💾 Saved → {output_csv}")
    print(df["Downstream6bp"].value_counts().head(10))

    return output_csv  # <-- pass this to next script


if __name__ == "__main__":
    path = input("Enter FASTQ file path: ")
    main(path)
