import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from skimage.io import imread
from skimage.draw import disk
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def inspect_decay_curve(stack, mask):
    sensor_curve = stack[mask].mean(axis=0)
    timepoints = np.arange(0, 20 * stack.shape[-1], 20)
    plt.plot(timepoints, sensor_curve, 'o-')
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean Intensity")
    plt.title("Average Decay Curve for Sensor")
    plt.grid(True)
    plt.show()



def exp_fit_func(V):
    H, W, n_ims = V.shape
    x_fit = np.arange(0, 20 * n_ims, 20)
    lifetime_map = np.zeros((H, W), dtype=np.float32)

    XX = np.vstack((np.ones(n_ims), x_fit)).T

    for i in range(H):
        for j in range(W):
            yy = V[i, j, :]
            if np.any(yy <= 0): continue  # skip bad pixels
            yy = np.clip(yy, 1e-3, None)
            lny = np.log(yy)

            try:
                betac, _, _, _ = np.linalg.lstsq(XX, lny, rcond=None)
                a = betac[1]  # slope
                if a <= 0 or abs(a) < 1e-6:
                    continue  # invalid or too flat
                lifetime_map[i, j] = -1.0 / a
            except Exception as e:
                continue  # skip failed fits

    # Clamp reasonable values
    lifetime_map[np.isnan(lifetime_map)] = 0
    lifetime_map[lifetime_map > 250] = 250
    lifetime_map[lifetime_map < 0] = 0
    return lifetime_map


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
    print("Click the 4 sensor centers. Close the window when done.")
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
    return [(x, y, radius) for x, y in points]

def analyze_condition_folder(folder_path, manual_circles, results_list):
    pgm_files = sorted(glob(os.path.join(folder_path, "*.pgm")))
    if len(pgm_files) < 11:
        print(f"Skipping {folder_path}: Not enough images.")
        return

    stack = np.stack([imread(f).astype(np.float32) for f in pgm_files[:11]], axis=-1)
    base_image = np.mean(stack, axis=-1)
    lifetime_map = exp_fit_func(stack)

    if not manual_circles:
        manual_circles.extend(manually_define_circles(base_image, radius=25))

    circles = manual_circles
    if len(circles) != 4:
        print(f"⚠️ Expected 4 sensor clicks but got {len(circles)}. Skipping {folder_path}")
        return

    radius = int(np.mean([r for _, _, r in circles]))
    centers = [(x, y) for x, y, _ in circles]
    masks = create_circle_mask(base_image, centers, radius)

    inspect_decay_curve(stack, masks[0])  # or masks[1], etc. to test others

    print(f"\n{os.path.basename(folder_path)}:")
    for i, mask in enumerate(masks):
        tau_values = lifetime_map[mask]
        avg_tau = np.mean(tau_values)
        std_tau = np.std(tau_values)

        masked_image = base_image * mask.astype(np.float32)
        nonzero_pixels = masked_image[masked_image > 0]

        if len(nonzero_pixels) == 0:
            avg_intensity = 0
            std_intensity = 0
        else:
            avg_intensity = np.mean(nonzero_pixels)
            std_intensity = np.std(nonzero_pixels)

        print(f" Sensor {i+1}: Avg τ = {avg_tau:.2f} ± {std_tau:.2f}, Avg Intensity = {avg_intensity:.2f}")
        results_list.append({
            "Folder": os.path.basename(folder_path),
            "Sensor": i + 1,
            "Avg_Tau": avg_tau,
            "Std_Tau": std_tau,
            "Avg_Intensity": avg_intensity,
            "Std_Intensity": std_intensity
        })

def analyze_all_conditions(main_dir):
    all_condition_folders = sorted([
        os.path.join(main_dir, d)
        for d in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, d))
    ])

    manual_circles = []
    results_list = []

    print(f"Found {len(all_condition_folders)} condition folders.")
    for folder in all_condition_folders:
        analyze_condition_folder(folder, manual_circles, results_list)

    df = pd.DataFrame(results_list)
    df.to_csv("lifetime_results.csv", index=False)
    print("\n✅ Results saved to lifetime_results.csv")

# ----------- MAIN EXECUTION ----------
if __name__ == "__main__":
    main_directory = ("/Users/timothyvernon/Documents/MATLAB/analyse_fibers/SpotTest_05_02_25_VFA/B1_Blank15mins_1gain")
    analyze_all_conditions(main_directory)

