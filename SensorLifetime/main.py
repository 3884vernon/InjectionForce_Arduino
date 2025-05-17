import os
import numpy as np
from sensor_detection import extract_intensity
from decay_fitting import fit_decay
from visualization import plot_sensor_lifetime_heatmap


def calculate_average_lifetime(folder_path):
    """Calculate average lifetime from multiple folders of images."""
    all_lifetimes = []
    sensor_lifetimes = {}  # To store lifetime for each sensor at a specific position

    # Loop through all subfolders in the main folder
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)

        if os.path.isdir(subfolder_path):
            print(f"Processing folder: {subfolder}")

            # Extract intensity data (time and intensity values)
            times, intensities = extract_intensity(subfolder_path)

            if len(times) > 0 and len(intensities) > 0:
                # Fit decay model and get the lifetime (tau)
                tau = fit_decay(times, intensities)

                # Store the lifetime for each sensor based on its position
                # Here, cx and cy represent the center coordinates of the sensor
                sensor_lifetimes[(cx, cy)] = tau  # You should get cx, cy from sensor detection logic

                all_lifetimes.append(tau)

    # Calculate and print the average lifetime
    if all_lifetimes:
        avg_lifetime = np.mean(all_lifetimes)
        print(f"Average Lifetime: {avg_lifetime:.2f} µs")

        # Visualize the heatmap of sensor locations and their lifetimes
        plot_sensor_lifetime_heatmap(folder_path, sensor_lifetimes)
    else:
        print("No valid data found.")


# Example usage
folder_path = '/Users/timothyvernon/Documents/SpotTest_05_02_25_VFA/B1_Blank15mins_1gain'  # Specify the root folder path
calculate_average_lifetime(folder_path)
