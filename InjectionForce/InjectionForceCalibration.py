import serial
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === USER SETTINGS ===
PORT = '/dev/tty.usbmodem4827E2E15FC02'  # Update to your actual port
BAUD_RATE = 9600
DURATION = 10  # seconds per weight
WEIGHTS_G = [50, 100, 150, 200]
GRAVITY = 9.80665  # m/s²

# === INIT SERIAL ===
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)

# === CALIBRATION DATA ===
forces_n = []
adc_avgs = []

for weight_g in WEIGHTS_G:
    input(f"\nAdd {weight_g}g to the sensor and press Enter to begin 10s recording...")
    force_n = weight_g / 1000 * GRAVITY
    forces_n.append(force_n)

    readings = []
    start_time = time.time()

    while time.time() - start_time < DURATION:
        raw = ser.readline().decode('utf-8', errors='ignore').strip()
        if "Force Signal =" in raw:
            try:
                adc = int(raw.split('=')[1].strip())
                readings.append(adc)
            except ValueError:
                continue

    avg_adc = np.mean(readings)
    adc_avgs.append(avg_adc)
    print(f"Collected {len(readings)} samples. Avg ADC = {avg_adc:.2f}")

# === LINEAR REGRESSION ===
X = np.array(forces_n).reshape(-1, 1)
y = np.array(adc_avgs)
model = LinearRegression().fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# === OUTPUT RESULTS ===
print("\n=== CALIBRATION RESULTS ===")
print(f"y = {slope:.3f} * x + {intercept:.3f}")
print(f"R² = {r2:.4f}")

# === OPTIONAL: PLOT ===
plt.figure()
plt.scatter(forces_n, adc_avgs, label='Measured Data', color='blue')
plt.plot(forces_n, y_pred, label=f'Fit: y = {slope:.2f}x + {intercept:.2f}', color='red')
plt.xlabel('Force (N)')
plt.ylabel('ADC Value')
plt.title('Force Sensor Calibration')
plt.legend()
plt.grid(True)
plt.show()

# === CLOSE SERIAL ===
ser.close()
