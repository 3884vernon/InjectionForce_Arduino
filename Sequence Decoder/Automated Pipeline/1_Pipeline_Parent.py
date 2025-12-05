#!/usr/bin/env python3
"""
Pipeline Master — Automates your full sequence decoding workflow:
1️⃣ Extracts variant regions from FASTQ (TargetRegion_FullLibrary.py)
2️⃣ Assigns barcodes & generates heatmap (Barcode_Well_Identification.py)
3️⃣ Translates variant DNA → amino acids by well (Well_AA_Translator.py)

Each module remains independent and debuggable.
"""

import os
import importlib

# === USER CONFIGURATION ===
# Define the directory and FASTQ filename once — all scripts share these.
DATA_DIR = "/Users/timothyvernon/Downloads"
FASTQ_FILE = "4GSY4B_1_sample_1.fastq"
BARCODE_FILE = "idt-master-index-list.xlsx"
SHEET_NAME = "xGen UDI Primers Plate 1"
MAX_MISMATCHES = 1
THREADS = 0  # 0 = auto
# ==========================


def run_script(module_name, *args, **kwargs):
    """Dynamically import and run each script’s main() function."""
    module = importlib.import_module(module_name)
    if hasattr(module, "main"):
        print(f"\n🚀 Running {module_name}.main() ...")
        result = module.main(*args, **kwargs)
        print(f"✅ {module_name} completed.")
        return result
    else:
        raise ImportError(f"❌ {module_name} has no main() function!")


def main():
    print("🔗 Starting full sequence decoding pipeline...\n")

    # --- Step 1: Extract target region from FASTQ ---
    print("🧬 STEP 1 — Extracting variant regions from FASTQ")
    TargetRegion_FullLibrary = importlib.import_module("TargetSeq_Lib")
    extracted_csv = TargetRegion_FullLibrary.main(
        fastq_path=os.path.join(DATA_DIR, FASTQ_FILE)
    )
    basename = TargetRegion_FullLibrary.basename
    print(f"📄 Output: {extracted_csv}")

    # --- Step 2: Barcode mapping ---
    print("\n🏷️ STEP 2 — Matching barcodes and generating heatmap")
    Barcode_Well_Identification = importlib.import_module("Well_BarcodeID")
    annotated_csv = Barcode_Well_Identification.main(
        input_csv=extracted_csv,
        barcode_file=os.path.join(DATA_DIR, BARCODE_FILE),
        sheet_name=SHEET_NAME,
        max_mismatches=MAX_MISMATCHES,
        threads=THREADS,
        basename=basename
    )
    print(f"📄 Output: {annotated_csv}")

    # --- Step 3: Amino acid translation ---
    print("\n🔡 STEP 3 — Translating variant sequences to amino acids")
    Well_AA_Translator = importlib.import_module("AA_WellID_Translator")
    aa_summary_csv = Well_AA_Translator.main(input_csv=annotated_csv, basename=basename)
    print(f"📄 Output: {aa_summary_csv}")

    print("\n🎯 All steps complete!")
    print(f"  1️⃣ Extracted variants: {extracted_csv}")
    print(f"  2️⃣ Barcode annotated:  {annotated_csv}")
    print(f"  3️⃣ AA translation:     {aa_summary_csv}")


if __name__ == "__main__":
    main()
