import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def plot_sensor_lifetime_heatmap(folder_path, sensor_lifetimes):
    """Generate a heatmap of sensor locations with their lifetimes."""
    # Load a sample image to get the image dimensions
    sample_image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    sample_image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
    height, width = sample_image.shape

    # Initialize an empty heatmap
    heatmap = np.zeros((height, width))

    # Populate the heatmap with the sensor lifetimes
    for sensor_position, lifetime in sensor_lifetimes.items():
        cx, cy = sensor_position
        heatmap[cy, cx] = lifetime

    # Plot the heatmap
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Lifetime (µs)")
    plt.title("Sensor Lifetime Heatmap")
    plt.show()
