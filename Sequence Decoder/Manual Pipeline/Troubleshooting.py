import os

import pandas as pd

dir = "/Users/timothyvernon/Downloads"
filename = "idt-master-index-list.xlsx"

targetfile = os.path.join(dir, filename)

# List all sheet names
xls = pd.ExcelFile(targetfile)


sheet_name = "xGen UDI Primers Plate 1"

# Skip the top 3 rows to reach the actual table header
barcodes = pd.read_excel(targetfile, sheet_name=sheet_name, skiprows=3)

# Clean up column names (they may contain spaces or NaN)
barcodes = barcodes.rename(columns=lambda x: str(x).strip())

# Show the first few columns to confirm structure
print("✅ Loaded barcode sheet:")
print(barcodes.head())
print(barcodes.columns)



print("Available sheets:", xls.sheet_names)

# Preview the third sheet (or directly by name)
sheet_name = "xGen UDI Primers Plate 1"
df = pd.read_excel(targetfile, sheet_name=sheet_name)
print(df.columns)
print(df.head())
