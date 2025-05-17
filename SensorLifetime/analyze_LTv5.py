import os
from os import close

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from glob import glob
from skimage.io import imread
from skimage.draw import disk
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.close('all')


def create_binary_mask(image, threshold_factor=0.5):
    """
    Create a binary mask based on intensity thresholding

    Parameters:
    -----------
    image : numpy.ndarray
        2D array representing the image
    threshold_factor : float
        Factor to determine threshold as a percentage of the image dynamic range

    Returns:
    --------
    mask : numpy.ndarray
        Binary mask where True indicates pixels to analyze
    """
    # Get image statistics
    img_min = np.percentile(image, 1)  # 1st percentile to avoid outliers
    img_max = np.percentile(image, 99)  # 99th percentile to avoid outliers

    # Calculate threshold based on the dynamic range
    threshold = img_min + threshold_factor * (img_max - img_min)

    # Create binary mask
    mask = image > threshold

    # Optional: Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask_uint8 = mask.astype(np.uint8) * 255
    mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

    return mask_cleaned > 0


def inspect_decay_curve(stack, mask, title="Average Decay Curve for Sensor"):
    """Visualize and return the decay curve for debugging"""
    if np.sum(mask) == 0:
        print(f"Warning: Empty mask in {title}")
        return np.zeros(stack.shape[-1]), np.arange(0, 20 * stack.shape[-1], 20)

    sensor_curve = stack[mask].mean(axis=0)
    timepoints = np.arange(0, 20 * stack.shape[-1], 20)
    plt.figure(figsize=(10, 6))
    plt.plot(timepoints, sensor_curve, 'o-')
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean Intensity")
    plt.title(title)
    plt.grid(True)
    plt.show()

    # Return for debugging
    return sensor_curve, timepoints


def check_for_increasing_signal(curve, timepoints, threshold=0.05):
    """
    Check if signal is increasing over time or not decaying properly

    Returns:
    --------
    is_increasing : bool
        True if signal appears to be increasing
    diagnosis : str
        Description of the issue
    """
    # Check if end of curve is higher than beginning
    start_avg = np.mean(curve[:3])
    end_avg = np.mean(curve[-3:])

    # Calculate percent decrease
    if start_avg <= 0:
        return True, "Signal starts at or below zero"

    percent_decrease = (start_avg - end_avg) / start_avg

    if end_avg > start_avg:
        return True, f"Signal increases: {start_avg:.1f} → {end_avg:.1f}"
    elif percent_decrease < threshold:
        return True, f"Minimal decay: only {percent_decrease * 100:.1f}% decrease"

    return False, "Normal decay pattern"


def improved_exp_fit_func(V, debug=False, mask=None, binary_mask=None, min_lifetime=10, max_lifetime=250):
    """
    Enhanced version with better handling of edge cases and debugging capabilities

    Parameters:
    -----------
    V : numpy.ndarray
        3D array of shape (H, W, n_ims) containing the decay curves
    debug : bool
        Whether to print debugging information
    mask : numpy.ndarray, optional
        Boolean mask to limit fitting to specific regions (e.g., sensor areas)
    binary_mask : numpy.ndarray, optional
        Boolean mask for pixels with sufficient signal quality
    min_lifetime : float
        Minimum acceptable lifetime value (ms)
    max_lifetime : float
        Maximum acceptable lifetime value (ms)

    Returns:
    --------
    lifetime_map : numpy.ndarray
        2D array of lifetime values
    metrics : dict
        Diagnostic metrics if debug=True
    """
    H, W, n_ims = V.shape
    x_fit = np.arange(0, 20 * n_ims, 20)
    lifetime_map = np.zeros((H, W), dtype=np.float32)
    certainty_map = np.zeros((H, W), dtype=np.float32)  # Store fitting quality

    # Combine masks if both are provided
    combined_mask = np.ones((H, W), dtype=bool)
    if mask is not None:
        combined_mask = np.logical_and(combined_mask, mask)
    if binary_mask is not None:
        combined_mask = np.logical_and(combined_mask, binary_mask)

    XX = np.vstack((np.ones(n_ims), x_fit)).T

    # Count fitting issues for debugging
    skip_negative = 0
    skip_flat = 0
    skip_error = 0
    total_pixels = 0
    successful_fits = 0
    low_intensity_skips = 0
    low_decay_skips = 0

    # Special debug for a specific masked area
    if debug and mask is not None:
        debug_values = []

    for i in range(H):
        for j in range(W):
            if not combined_mask[i, j]:
                continue

            total_pixels += 1
            yy = V[i, j, :]

            # Calculate intensity stats
            peak_intensity = np.max(yy)
            avg_intensity = np.mean(yy)

            # Skip very low intensity pixels
            if avg_intensity < 10:  # Adjust threshold based on your data
                low_intensity_skips += 1
                continue

            # Calculate SNR for this pixel's curve
            signal_level = np.mean(yy)
            noise_level = np.std(np.diff(yy)) / np.sqrt(2)  # Estimate noise from differences
            snr = signal_level / (noise_level + 1e-10)  # Avoid division by zero

            # Skip pixels with very low SNR
            if snr < 3:  # Threshold can be adjusted
                skip_negative += 1
                continue

            # Check if curve is actually decaying
            start_avg = np.mean(yy[:3])
            end_avg = np.mean(yy[-3:])
            decay_percent = (start_avg - end_avg) / (start_avg + 1e-10)

            if decay_percent < 0.1:  # Less than 10% decay over the measurement
                low_decay_skips += 1
                continue

            # Skip pixels with any non-positive values
            if np.any(yy <= 0):
                skip_negative += 1
                continue

            # Clip very small values to avoid log issues
            yy = np.clip(yy, 1e-3, None)
            lny = np.log(yy)

            try:
                # Linear fit to log data
                betac, residuals, rank, s = np.linalg.lstsq(XX, lny, rcond=None)
                a = betac[1]  # slope

                # Calculate R² to check fit quality
                y_mean = np.mean(lny)
                ss_tot = np.sum((lny - y_mean) ** 2)
                if ss_tot == 0:  # Avoid division by zero
                    r_squared = 0
                else:
                    ss_res = residuals[0] if len(residuals) > 0 else np.sum((lny - XX @ betac) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)

                # Higher threshold for R²
                if r_squared < 0.7:  # Adjust based on your data quality
                    skip_flat += 1
                    continue

                # Skip if slope is positive (indicates increasing signal, not decay)
                if a >= 0:
                    skip_flat += 1
                    continue

                # Calculate lifetime from slope
                lifetime = -1.0 / a

                # Check if lifetime is in acceptable range
                if lifetime < min_lifetime or lifetime > max_lifetime:
                    skip_flat += 1
                    continue

                # Store lifetime and fit quality
                lifetime_map[i, j] = lifetime
                certainty_map[i, j] = r_squared
                successful_fits += 1

                # Store values for debugging a specific masked region
                if debug and mask is not None and mask[i, j]:
                    debug_values.append((i, j, a, lifetime, r_squared, snr))

            except Exception as e:
                skip_error += 1
                continue

    # Output diagnostics
    if debug:
        print(f"\nFitting Debug Info:")
        print(f"Total pixels processed: {total_pixels}")
        print(f"Successful fits: {successful_fits} ({successful_fits / max(1, total_pixels) * 100:.1f}%)")
        print(
            f"Skipped due to low intensity: {low_intensity_skips} ({low_intensity_skips / max(1, total_pixels) * 100:.1f}%)")
        print(
            f"Skipped due to insufficient decay: {low_decay_skips} ({low_decay_skips / max(1, total_pixels) * 100:.1f}%)")
        print(f"Skipped due to negative/low SNR: {skip_negative} ({skip_negative / max(1, total_pixels) * 100:.1f}%)")
        print(f"Skipped due to flat/poor fits: {skip_flat} ({skip_flat / max(1, total_pixels) * 100:.1f}%)")
        print(f"Skipped due to errors: {skip_error} ({skip_error / max(1, total_pixels) * 100:.1f}%)")

        if mask is not None and debug_values:
            slopes = [v[2] for v in debug_values]
            lifetimes = [v[3] for v in debug_values]
            r_squared_values = [v[4] for v in debug_values]
            snr_values = [v[5] for v in debug_values]

            print(f"Masked region stats:")
            print(f"  Number of fitted pixels: {len(debug_values)}")
            print(f"  Average slope: {np.mean(slopes):.6f}")
            print(f"  Average lifetime: {np.mean(lifetimes):.2f}")
            print(f"  Average R²: {np.mean(r_squared_values):.3f}")
            print(f"  Average SNR: {np.mean(snr_values):.1f}")

            # Display example fits for a few points
            if debug_values:
                print("\nExample fits for 3 random pixels in mask:")
                import random
                samples = min(3, len(debug_values))
                for idx in random.sample(range(len(debug_values)), samples):
                    i, j, slope, lt, r2, snr = debug_values[idx]
                    y_vals = V[i, j, :]
                    log_y = np.log(np.clip(y_vals, 1e-3, None))

                    # Get the betac for this specific pixel
                    betac, _, _, _ = np.linalg.lstsq(XX, log_y, rcond=None)
                    fitted_line = XX @ betac

                    print(f"Pixel ({i},{j}): slope={slope:.6f}, lifetime={lt:.2f}, R²={r2:.3f}, SNR={snr:.1f}")
                    print(f"  Original values: {y_vals}")
                    print(f"  Log values: {log_y}")

                    plt.figure(figsize=(10, 6))
                    plt.subplot(1, 2, 1)
                    plt.plot(x_fit, y_vals, 'o-', label='Original Data')
                    plt.plot(x_fit, np.exp(fitted_line), 'r-', label=f'Fit (τ={lt:.2f}ms)')
                    plt.title(f"Pixel ({i},{j}) - Intensity vs Time")
                    plt.xlabel("Time (ms)")
                    plt.ylabel("Intensity")
                    plt.legend()
                    plt.grid(True)

                    plt.subplot(1, 2, 2)
                    plt.plot(x_fit, log_y, 'o', label='Log Data')
                    plt.plot(x_fit, fitted_line, 'r-', label=f'Fit (R²={r2:.3f})')
                    plt.title(f"Log Plot - Lifetime = {lt:.2f}ms")
                    plt.xlabel("Time (ms)")
                    plt.ylabel("Log(Intensity)")
                    plt.legend()
                    plt.grid(True)

                    plt.tight_layout()
                    plt.show()

    return (lifetime_map, certainty_map) if debug else lifetime_map


def visualize_enhanced_results(base_image, lifetime_map, certainty_map, masks, binary_mask=None):
    """
    Visualize results with enhanced debugging and certainty information
    """
    from matplotlib.colors import LinearSegmentedColormap
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    thermal_cmap = LinearSegmentedColormap.from_list(
        "blue_cyan_green_yellow_red",
        ["blue", "cyan", "green", "yellow", "red"],
        N=256
    )

    plt.figure(figsize=(18, 12))

    # Show the base image
    plt.subplot(2, 3, 1)
    plt.imshow(base_image, cmap='gray')
    plt.title('Base Image')
    plt.colorbar()

    # Draw circles on image
    colored_img = cv2.cvtColor(cv2.normalize(base_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                               cv2.COLOR_GRAY2BGR)
    for i, mask in enumerate(masks):
        # Find center of mask
        indices = np.where(mask)
        if len(indices[0]) > 0:
            center_y = int(np.mean(indices[0]))
            center_x = int(np.mean(indices[1]))
            radius = int(np.sqrt(len(indices[0]) / np.pi))
            cv2.circle(colored_img, (center_x, center_y), radius, (0, 0, 255), 2)
            cv2.putText(colored_img, f"{i + 1}", (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB))
    plt.title('Sensors')

    # Show binary mask if provided
    if binary_mask is not None:
        plt.subplot(2, 3, 3)
        plt.imshow(binary_mask, cmap='gray')
        plt.title('Binary Mask')
        plt.colorbar()

    # Show the lifetime map
    plt.subplot(2, 3, 4)
    lifetime_viz = np.copy(lifetime_map)
    plt.imshow(lifetime_viz, cmap=thermal_cmap, vmin=0, vmax=250)
    plt.title('Lifetime Map')
    plt.colorbar(label='Lifetime (ms)')

    # Show the certainty map (R²)
    plt.subplot(2, 3, 5)
    plt.imshow(certainty_map, cmap='viridis', vmin=0, vmax=1)
    plt.title('Fit Quality (R²)')
    plt.colorbar(label='R²')

    # Show overlap of binary mask and lifetime
    if binary_mask is not None:
        plt.subplot(2, 3, 6)
        masked_lifetime = np.ma.masked_where(lifetime_map <= 0, lifetime_map)
        plt.imshow(binary_mask, cmap='gray', alpha=0.3)
        plt.imshow(masked_lifetime, cmap=thermal_cmap, vmin=0, vmax=250, alpha=0.7)
        plt.title('Mask + Lifetime Overlay')
        plt.colorbar(label='Lifetime (ms)')

    plt.tight_layout()
    plt.show()

    # Create a histogram of lifetime values
    plt.figure(figsize=(12, 6))

    # For each sensor, plot histogram
    nonzero_lifetimes = lifetime_map[lifetime_map > 0]
    if len(nonzero_lifetimes) > 0:
        plt.subplot(1, 2, 1)
        plt.hist(nonzero_lifetimes, bins=50, alpha=0.7)
        plt.xlabel('Lifetime (ms)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Lifetimes')
        plt.grid(True)

        # Add plot of lifetime vs certainty
        plt.subplot(1, 2, 2)
        # Flatten arrays
        flat_lifetime = lifetime_map.flatten()
        flat_certainty = certainty_map.flatten()
        # Only plot non-zero lifetime values
        valid_idx = flat_lifetime > 0
        plt.scatter(flat_lifetime[valid_idx], flat_certainty[valid_idx], alpha=0.3, s=1)
        plt.xlabel('Lifetime (ms)')
        plt.ylabel('Fit Quality (R²)')
        plt.title('Lifetime vs. Fit Quality')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


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


def analyze_condition_folder(folder_path, manual_circles, results_list, debug=True, binary_threshold=0.3,
                             min_lifetime=10, max_lifetime=250):
    pgm_files = sorted(glob(os.path.join(folder_path, "*.pgm")))
    if len(pgm_files) < 11:
        print(f"Skipping {folder_path}: Not enough images.")
        return

    print(f"\nAnalyzing {os.path.basename(folder_path)}...")

    # Load the image stack
    stack = np.stack([imread(f).astype(np.float32) for f in pgm_files[:11]], axis=-1)
    base_image = np.mean(stack, axis=-1)

    # Debug: Check overall stack stats
    if debug:
        print(f"Stack shape: {stack.shape}")
        print(f"Stack min: {np.min(stack)}, max: {np.max(stack)}, mean: {np.mean(stack):.2f}")

    # Create binary mask based on intensity
    binary_mask = create_binary_mask(base_image, threshold_factor=binary_threshold)
    if debug:
        print(f"Binary mask coverage: {np.sum(binary_mask) / binary_mask.size * 100:.2f}% of image")

    # Get or define circles
    if not manual_circles:
        manual_circles.extend(manually_define_circles(base_image, radius=25))

    circles = manual_circles
    if len(circles) != 4:
        print(f"⚠️ Expected 4 sensor clicks but got {len(circles)}. Skipping {folder_path}")
        return

    radius = int(np.mean([r for _, _, r in circles]))
    centers = [(x, y) for x, y, _ in circles]
    masks = create_circle_mask(base_image, centers, radius)

    # Inspect decay curve for debugging
    print("\nInspecting decay curves for each sensor:")
    for i, mask in enumerate(masks):
        # Combine with binary mask
        combined_mask = np.logical_and(mask, binary_mask)
        if np.sum(combined_mask) == 0:
            print(f"⚠️ WARNING: No pixels in combined mask for Sensor {i + 1}")
            continue

        curve, times = inspect_decay_curve(stack, combined_mask, f"Decay curve for Sensor {i + 1}")

        # Check decay patterns
        is_increasing, diagnosis = check_for_increasing_signal(curve, times)
        if is_increasing:
            print(f"⚠️ WARNING: Sensor {i + 1} - {diagnosis}")

    # Calculate lifetime map with debugging
    print("\nCalculating lifetime map...")
    lifetime_map, certainty_map = improved_exp_fit_func(stack, debug=True, binary_mask=binary_mask,
                                                        min_lifetime=min_lifetime, max_lifetime=max_lifetime)

    # Visualize results
    if debug:
        visualize_enhanced_results(base_image, lifetime_map, certainty_map, masks, binary_mask)

    # Additional debug: Check lifetime map stats
    if debug:
        print(f"\nLifetime map stats:")
        print(
            f"  Min: {np.min(lifetime_map)}, Max: {np.max(lifetime_map)}, Mean: {np.mean(lifetime_map[lifetime_map > 0]):.2f}")
        print(f"  % Non-zero pixels: {np.count_nonzero(lifetime_map) / lifetime_map.size * 100:.2f}%")

    # Debug detailed analysis for each mask
    print(f"\n{os.path.basename(folder_path)}:")
    for i, mask in enumerate(masks):
        # Combine with binary mask for analysis
        combined_mask = np.logical_and(mask, binary_mask)

        # Debug: Calculate lifetime map specifically for this mask
        masked_lifetime, _ = improved_exp_fit_func(stack, debug=True, mask=combined_mask,
                                                   min_lifetime=min_lifetime, max_lifetime=max_lifetime)

        tau_values = lifetime_map[combined_mask]
        nonzero_tau = tau_values[tau_values > 0]

        if len(nonzero_tau) > 0:
            avg_tau = np.mean(nonzero_tau)
            std_tau = np.std(nonzero_tau)
            valid_percent = len(nonzero_tau) / max(1, len(tau_values)) * 100
        else:
            avg_tau = 0
            std_tau = 0
            valid_percent = 0

        masked_image = base_image * combined_mask.astype(np.float32)
        nonzero_pixels = masked_image[masked_image > 0]

        if len(nonzero_pixels) == 0:
            avg_intensity = 0
            std_intensity = 0
        else:
            avg_intensity = np.mean(nonzero_pixels)
            std_intensity = np.std(nonzero_pixels)

        print(f" Sensor {i + 1}: Avg τ = {avg_tau:.2f} ± {std_tau:.2f} ms ({valid_percent:.1f}% valid), "
              f"Avg Intensity = {avg_intensity:.2f}")

        # Check for issues with decay curve
        curve, _ = inspect_decay_curve(stack, combined_mask, f"Final decay curve check - Sensor {i + 1}")
        is_increasing, diagnosis = check_for_increasing_signal(curve, np.arange(0, 20 * len(curve), 20))
        if is_increasing:
            print(f"   ⚠️ Note: {diagnosis} - May explain low valid pixel percentage")

        results_list.append({
            "Folder": os.path.basename(folder_path),
            "Sensor": i + 1,
            "Avg_Tau": avg_tau,
            "Std_Tau": std_tau,
            "Avg_Intensity": avg_intensity,
            "Std_Intensity": std_intensity,
            "Valid_Percent": valid_percent,
            "Decay_Diagnosis": diagnosis
        })


def analyze_all_conditions(main_dir, debug=True, binary_threshold=0.3, min_lifetime=10, max_lifetime=250):
    all_condition_folders = sorted([
        os.path.join(main_dir, d)
        for d in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, d))
    ])

    manual_circles = []
    results_list = []

    print(f"Found {len(all_condition_folders)} condition folders.")
    for folder in all_condition_folders:
        analyze_condition_folder(folder, manual_circles, results_list, debug=debug,
                                 binary_threshold=binary_threshold,
                                 min_lifetime=min_lifetime,
                                 max_lifetime=max_lifetime)

    df = pd.DataFrame(results_list)
    df.to_csv("lifetime_results.csv", index=False)
    print("\n✅ Results saved to lifetime_results.csv")

    # Visualize results across all conditions
    if debug and len(results_list) > 0:
        plt.figure(figsize=(12, 8))

        # Group by folder and sensor
        df_plot = df.pivot(index='Folder', columns='Sensor', values='Avg_Tau')
        df_plot.plot(kind='bar', yerr=df.pivot(index='Folder', columns='Sensor', values='Std_Tau'),
                     capsize=5, figsize=(12, 8))

        plt.title('Average Lifetime by Condition and Sensor')
        plt.ylabel('Lifetime (ms)')
        plt.xlabel('Condition')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

        # Plot valid pixel percentage
        plt.figure(figsize=(12, 8))
        df.pivot(index='Folder', columns='Sensor', values='Valid_Percent').plot(kind='bar', figsize=(12, 8))
        plt.title('Valid Pixel Percentage by Condition and Sensor')
        plt.ylabel('Valid Pixel %')
        plt.xlabel('Condition')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()


# Test the exponential fit with a known curve
def test_exp_fit():
    print("\n==== Testing Exponential Fit Function ====")
    # Create synthetic data: 50x50 image with exponential decay
    x_vals = np.arange(0, 11)  # 11 time points

    # True lifetime = 200ms
    true_lifetime = 200
    decay_rate = -1 / true_lifetime

    # Create simple decay curve: y = A*exp(decay_rate * t)
    A = 1000  # initial amplitude
    decay_curve = A * np.exp(decay_rate * x_vals * 20)  # 20ms intervals

    print(f"True decay rate: {decay_rate}")
    print(f"True lifetime: {true_lifetime} ms")
    print(f"Generated curve: {decay_curve}")

    # Create a small 50x50x11 test volume
    test_vol = np.zeros((50, 50, 11))

    # Add noise to make it realistic
    noise_level = 0.05  # 5% noise

    for i in range(50):
        for j in range(50):
            # Add random variation to amplitude
            amp_var = A * (1 + np.random.uniform(-0.1, 0.1))
            noise = noise_level * amp_var * np.random.normal(0, 1, 11)
            test_vol[i, j, :] = amp_var * np.exp(decay_rate * x_vals * 20) + noise

    # Create a simple mask in the center
    test_mask = np.zeros((50, 50), dtype=bool)
    test_mask[15:35, 15:35] = True

    # Create a binary mask
    binary_mask = create_binary_mask(np.mean(test_vol, axis=2), threshold_factor=0.3)

    # Visualize test curve
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals * 20, decay_curve, 'o-', label='True decay curve')
    plt.plot(x_vals * 20, test_vol[25, 25, :], 'x-', label='With noise')
    plt.xlabel("Time (ms)")
    plt.ylabel("Intensity")
    plt.title(f"Test Decay Curve (τ = {true_lifetime} ms)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Test the function
    test_lifetime_map, test_certainty_map = improved_exp_fit_func(test_vol, debug=True, binary_mask=binary_mask)

    # Show results
    plt.figure(figsize=(10, 6))
    plt.imshow(test_lifetime_map, cmap='viridis', vmin=0, vmax=250)
    plt.colorbar(label='Lifetime (ms)')
    plt.title('Test Lifetime Map')
    plt.show()

    # Calculate average in test region
    test_values = test_lifetime_map[test_mask]
    nonzero_test = test_values[test_values > 0]

    if len(nonzero_test) > 0:
        avg_test = np.mean(nonzero_test)
        std_test = np.std(nonzero_test)
        print(f"Recovered lifetime: {avg_test:.2f} ± {std_test:.2f} ms")
        print(f"Error: {100 * abs(avg_test - true_lifetime) / true_lifetime:.2f}%")
    else:
        print("No valid lifetime values recovered in test!")

    return decay_curve, test_lifetime_map


# ----------- MAIN EXECUTION ----------
if __name__ == "__main__":
    # First run a test with synthetic data
    test_exp_fit()

    # Then analyze the real data
    main_directory = ("/Users/timothyvernon/Documents/PhD/Projects/gMAP/SpotTest/SpotTest_05_02_25_VFA/B3_250TLBR_0TRBL_1gain")
    analyze_all_conditions(main_directory, debug=True)