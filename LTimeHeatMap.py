import cv2
import numpy as np
import os
import scipy.optimize as opt


# Function to find the sensor location in the image
def find_sensor_location(image):
    # For simplicity, assuming sensors are circular and we can detect them via thresholding
    gray = image
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find contours of the thresholded image, assuming the sensor is circular
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there are contours, assume the first one is the sensor
    if contours:
        sensor_contour = max(contours, key=cv2.contourArea)
        # Get the center of the sensor (assumed to be circular)
        M = cv2.moments(sensor_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
    return None


# Function to extract the intensity over time for the sensor
def extract_intensity(image_folder):
    intensities = []
    times = []

#Time step in microseconds starting from 0us to 200us
    time_step = 20 #20us per image

    # List of images in the folder (assuming filenames are time-dependent)
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.pgm')])

    for idx, file in enumerate(image_files):
        image_path = os.path.join(image_folder, file)
        # Read the image in grayscale for .pgm files
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Error: Could not read image {image_path}")
            continue

        # Find the sensor location
        sensor_location = find_sensor_location(image)

        if sensor_location:
            cx, cy = sensor_location

            # Extract a small region around the sensor to measure intensity (e.g., 10x10 px)
            intensity_region = image[cy - 5:cy + 5, cx - 5:cx + 5]
            intensity = np.mean(intensity_region)

            intensities.append(intensity)
            times.append(idx*time_step)  # assuming the images are time-sequenced

    # Debugging - Check the first 11 intensity values and first 5 Tau values
    print(f"Intensity values: {intensities[:10]}")  # Print the first 11 intensity values
    print(f"Time values: {times[:11]}")  # Print the first 11 time values

    return times, intensities


# Decay model fitting (exponential decay)
def decay_model(t, I0, tau):
    return I0 * np.exp(-t / tau)


# Fit the decay model to the intensity data
def fit_decay(times, intensities):
    # Initial guess for parameters
    initial_guess = [np.max(intensities), 1.0]

    # Fit the data to the exponential decay model
    params, covariance = opt.curve_fit(decay_model, times, intensities, p0=initial_guess)
    I0, tau = params

    return tau  # Lifetime constant


# Main function to calculate average lifetime from multiple folders of images
def calculate_average_lifetime(folder_path):
    all_lifetimes = []

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
                all_lifetimes.append(tau)

    # Calculate the average lifetime
    if all_lifetimes:
        avg_lifetime = np.mean(all_lifetimes)
        print(f"Average Lifetime: {avg_lifetime:.2f} units")
    else:
        print("No valid data found.")


# Example usage
folder_path = '/Users/timothyvernon/Documents/SpotTest_05_02_25_VFA/B1_Blank15mins_50gain'  # Specify the root folder path
calculate_average_lifetime(folder_path)


