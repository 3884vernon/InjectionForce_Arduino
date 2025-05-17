import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

from pyparsing import results
from skimage.io import imread
from skimage.draw import disk
from skimage.util import img_as_float
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def exp_fit_func(V):
    H, W, n_ims = V.shape
    x_fit = np.arange(0, 20 * n_ims, 20)
    lifetime_map = np.zeros((H, W), dtype=np.float32)
    init_map = np.zeros((H, W), dtype=np.float32)
    score_rsq = np.zeros((H, W), dtype=np.float32)
    score_rmse = np.zeros((H, W), dtype=np.float32)
    XX = np.vstack((np.ones(n_ims), x_fit)).T

    for i in range(H):
        for j in range(W):
            yy = V[i, j, :]
            if np.any(yy <= 0): continue
            lny = np.log(yy)
            betac, _, _, _ = np.linalg.lstsq(XX, lny, rcond=None)
            b, a = betac
            lifetime_map[i, j] = a
            init_map[i, j] = b
            y_calc = XX @ betac
            ss_res = np.sum((lny - y_calc) ** 2)
            ss_tot = np.sum((lny - np.mean(lny)) ** 2)
            rsq = 1 - ss_res / ss_tot if ss_tot != 0 else 0
            rmse = np.sqrt(np.sum((np.exp(y_calc) - yy) ** 2))
            score_rsq[i, j] = rsq
            score_rmse[i, j] = rmse

    LTM0 = -1.0 / lifetime_map
    LTM0[np.isnan(LTM0)] = 250
    LTM0[LTM0 > 250] = 250
    LTM0[LTM0 < 0] = 0
    return LTM0, init_map, score_rsq, score_rmse

def create_circle_mask(image, centers, radius):
    masks = []
    h, w = image.shape
    for center in centers:
        mask = np.zeros((h, w), dtype=bool)
        rr, cc = disk((center[1], center[0]), radius, shape=image.shape)
        mask[rr, cc] = True
        masks.append(mask)
    return masks


def manually_define_circles(image, radius=25):
    print("Click the 4 sensor centers (in consistent order). Close window when done.")
    img_display = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_img = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(color_img, (x, y), radius, (0, 0, 255), 2)
            cv2.imshow("Click Sensor Centers", color_img)

    cv2.imshow("Click Sensor Centers", color_img)
    cv2.setMouseCallback("Click Sensor Centers", mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Add radius to each point
    return [(x, y, radius) for x, y in points]

def analyze_condition_folder(folder_path, manual_circles, results_list):
    pgm_files = sorted(glob(os.path.join(folder_path, "*.pgm")))
    if len(pgm_files) < 11:
        print(f"Skipping {folder_path}: Not enough images.")
        return
    stack = np.stack([img_as_float(imread(f)) for f in pgm_files[:11]], axis=-1)
    base_image = stack[:, :, 0]
    lifetime_map, _, _, _ = exp_fit_func(stack)

    if not manual_circles:  # empty list = False in Python
        manual_circles.extend(manually_define_circles(base_image, radius=25))

    circles = manual_circles

    if len(circles) != 4:
        print(f"⚠️ Warning: Expected 4 sensor clicks but got {len(circles)}. Skipping {folder_path}")
        return

    radius = int(np.mean([r for _, _, r in circles]))
    centers = [(x, y) for x, y, _ in circles]
    masks = create_circle_mask(base_image, centers, radius)

    print(f"\n{os.path.basename(folder_path)}:")
    for i, mask in enumerate(masks):
        tau_values = lifetime_map[mask]
        intensity_values = base_image[mask]
        print(f" Sensor {i+1}: Avg τ = {np.mean(tau_values):.2f} ± {np.std(tau_values):.2f}, Avg Intensity = {np.mean(intensity_values):.2f}")

    # Show mask and heatmap
    mask_sum = np.sum([m.astype(np.uint8) for m in masks], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(mask_sum, cmap='gray')
    axes[0].set_title("Binary Mask for Sensors")
    im = axes[1].imshow(lifetime_map, cmap='hot')
    axes[1].set_title("Lifetime Heatmap")
    plt.colorbar(im, ax=axes[1], label='Lifetime (τ)')
    plt.tight_layout()
    plt.show()

def analyze_all_conditions(main_dir):
    all_condition_folders = sorted([os.path.join(main_dir, d)
                                    for d in os.listdir(main_dir)
                                    if os.path.isdir(os.path.join(main_dir, d))])

    manual_circles = []
    results_list = []

    print(f"Found {len(all_condition_folders)} condition folders.")
    for folder in all_condition_folders:
        analyze_condition_folder(folder,manual_circles, results_list)

# ----------- MAIN EXECUTION ----------
if __name__ == "__main__":
    # 🔧 Replace this with the path to your main directory
    main_directory = "/Users/timothyvernon/Documents/MATLAB/analyse_fibers/SpotTest_05_02_25_VFA/B3_Blank15mins_1gain"
    analyze_all_conditions(main_directory)


