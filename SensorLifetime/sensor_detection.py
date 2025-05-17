import os
import cv2
import numpy as np


def find_sensor_location(image):
    """Detect the sensor location in the image."""
    gray = image  # Already grayscale for .pgm images
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find contours of the thresholded image, assuming the sensor is circular
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        sensor_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(sensor_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
    return None


def extract_intensity(image_folder):
    """Extract intensity over time for the sensor across multiple images."""
    intensities = []
    times = []
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.pgm')])
    time_step = 20  # Time step of 20us per image

    for idx, file in enumerate(image_files):
        image_path = os.path.join(image_folder, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Error: Could not read image {image_path}")
            continue

        # Find the sensor location
        sensor_location = find_sensor_location(image)

        if sensor_location:
            cx, cy = sensor_location
            intensity_region = image[cy - 5:cy + 5, cx - 5:cx + 5]  # Small region around the sensor
            intensity = np.mean(intensity_region)
            intensities.append(intensity)
            times.append(idx * time_step)  # Adding 20 µs per image
            print(cx, cy, intensity)

    return times, intensities
