import cv2
import numpy as np
import os
import scipy.optimize as opt


# Function to visualize the heatmap of sensor locations and their lifetimes
def plot_sensor_lifetime_heatmap(folder_path, sensor_lifetimes):
    # Create a blank canvas for heatmap (based on image dimensions)
    # Assuming all images are the same size, take the size of the first image
    sample_image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    sample_image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
    height, width = sample_image.shape

    # Initialize an empty heatmap array
    heatmap = np.zeros((height, width))

    # Add the lifetimes to the heatmap based on sensor positions
    for sensor_position, lifetime in sensor_lifetimes.items():
        cx, cy = sensor_position
        # Update the heatmap at the sensor's location
        heatmap[cy, cx] = lifetime

    # Plot the heatmap
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Lifetime (µs)")
    plt.title("Sensor Lifetime Heatmap")
    plt.show()


# Main function to calculate average lifetime from multiple folders of images
def calculate_average_lifetime(folder_path):
    all_lifetimes = []
    sensor_lifetimes = {}  # To store lifetime for each sensor at a specific position

    # Loop through folders
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)

        if os.path.isdir(subfolder_path):
            print(f"Processing folder: {subfolder}")

            # Extract intensity data
            times, intensities = extract_intensity(subfolder_path)

            if len(times) > 0 and len(intensities) > 0:
                # Fit decay model and get the lifetime (tau)
                tau = fit_decay(times, intensities)

                # Store the lifetime for each sensor based on its position
                # Here we assume the sensor's position is at the center of the region we're extracting
                sensor_lifetimes[(cx, cy)] = tau  # You should adjust the (cx, cy) from sensor detection

                all_lifetimes.append(tau)

    # Calculate the average lifetime
    if all_lifetimes:
        avg_lifetime = np.mean(all_lifetimes)
        print(f"Average Lifetime: {avg_lifetime:.2f} µs")

        # Plot heatmap of sensor locations and lifetimes
        plot_sensor_lifetime_heatmap(folder_path, sensor_lifetimes)
    else:
        print("No valid data found.")


# Example usage
folder_path = '/Users/timothyvernon/Documents/SpotTest_05_02_25_VFA/B3_Blank15mins_1gain'  # Specify the root folder path
calculate_average_lifetime(folder_path)
