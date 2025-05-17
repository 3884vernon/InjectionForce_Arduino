import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from glob import glob
import os
from skimage.draw import disk

def run_intensity_diagnostics(folder_path):
    pgm_files = sorted(glob(os.path.join(folder_path, "*.pgm")))
    if len(pgm_files) == 0:
        print("❌ No .pgm files found in folder:", folder_path)
        return

    # Load the first image
    first_img = imread(pgm_files[0])
    print("✅ Loaded first image:", pgm_files[0])
    print(" - Shape:", first_img.shape)
    print(" - Dtype:", first_img.dtype)
    print(" - Min pixel value:", np.min(first_img))
    print(" - Max pixel value:", np.max(first_img))
    print(" - Unique pixel values (sample):", np.unique(first_img)[:10])

    # Show raw image
    plt.figure(figsize=(6, 5))
    plt.imshow(first_img, cmap='gray')
    plt.title("Raw First Image")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # Create test circular mask in image center
    center = (first_img.shape[1] // 2, first_img.shape[0] // 2)
    mask = np.zeros_like(first_img, dtype=bool)
    rr, cc = disk((center[1], center[0]), 40, shape=first_img.shape)
    mask[rr, cc] = True

    masked = first_img * mask
    nonzero_masked = masked[masked > 0]

    print("\n🔍 Mask Diagnostics:")
    print(" - Total pixels in mask:", np.count_nonzero(mask))
    print(" - Non-zero pixels in mask:", np.count_nonzero(nonzero_masked))
    print(" - Mean intensity (masked):", np.mean(nonzero_masked) if nonzero_masked.size else 0)
    print(" - Std intensity (masked):", np.std(nonzero_masked) if nonzero_masked.size else 0)

    # Show masked region
    plt.figure(figsize=(6, 5))
    plt.imshow(masked, cmap='gray')
    plt.title("Masked Region at Image Center")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

run_intensity_diagnostics("/Users/timothyvernon/Documents/MATLAB/analyse_fibers/SpotTest_05_02_25_VFA/B3_250TLBR_0TRBL_1gain/0153")
