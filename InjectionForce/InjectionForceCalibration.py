import serial
import time
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === USER SETTINGS ===
PORT = '/dev/tty.usbmodem4827E2E15FC02'
BAUD_RATE = 9600
DURATION = 10  # seconds of data collection
SETTLING_TIME = 2
WEIGHTS_G = [50, 100, 150, 200]
GRAVITY = 9.80665  # m/s²

# === INIT SERIAL ===
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)

# === OUTPUT DIR SETUP ===
output_dir = "calibration_outputs"
os.makedirs(output_dir, exist_ok=True)

forces_n = []
adc_avgs = []

for weight_g in WEIGHTS_G:
    input(f"\nAdd {weight_g}g to the sensor and press Enter to begin...")
    force_n = weight_g / 1000 * GRAVITY

    # CSV logging
    csv_filename = os.path.join(output_dir, f"adc_raw_{weight_g}g.csv")
    csv_file = open(csv_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Time (s)", "ADC"])

    readings = []
    timestamps = []

    # Stabilize
    print("Stabilizing sensor for 2 seconds...")
    ser.reset_input_buffer()
    time.sleep(SETTLING_TIME)

    # Setup Live Plot
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='Live ADC')
    ax.set_xlim(0, DURATION)
    ax.set_ylim(0, 1024)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ADC Value')
    ax.set_title(f'Live ADC Readings - {weight_g}g')
    ax.grid(True)
    fig.canvas.draw()

    print("Collecting data...")

    start_time = time.time()

    while time.time() - start_time < DURATION:
        raw = ser.readline().decode('utf-8', errors='ignore').strip()
        current_time = time.time() - start_time
        if "Force Signal =" in raw:
            try:
                adc = int(raw.split('=')[1].strip())
                readings.append(adc)
                timestamps.append(current_time)
                csv_writer.writerow([current_time, adc])
                print(f"\rLive ADC: {adc}", end="")

                line.set_xdata(timestamps)
                line.set_ydata(readings)
                ax.set_xlim(0, max(DURATION, current_time + 1))
                ax.set_ylim(min(readings[-50:] + [0]), max(readings[-50:] + [1023]))
                fig.canvas.draw()
                fig.canvas.flush_events()

            except ValueError:
                continue

    plt.ioff()
    plot_filename = os.path.join(output_dir, f"live_plot_{weight_g}g.png")
    plt.savefig(plot_filename)
    plt.close()
    csv_file.close()

    if len(readings) == 0:
        print(f"\n⚠️ No samples collected for {weight_g}g — skipping this point.")
        continue

    avg_adc = np.mean(readings)
    adc_avgs.append(avg_adc)
    forces_n.append(force_n)
    print(f"\nCollected {len(readings)} samples. Avg ADC = {avg_adc:.2f}")

# === CHECK IF ANY VALID DATA ===
if len(adc_avgs) == 0:
    print("❌ No valid data points collected. Exiting.")
    ser.close()
    exit(1)

# === REGRESSION ===
X = np.array(forces_n).reshape(-1, 1)
y = np.array(adc_avgs)
model = LinearRegression().fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# === RESULTS ===
print("\n=== CALIBRATION RESULTS ===")
print(f"y = {slope:.3f} * x + {intercept:.3f}")
print(f"R² = {r2:.4f}")

# === FINAL PLOT ===
plt.figure()
plt.scatter(forces_n, adc_avgs, label='Measured Data', color='blue')
plt.plot(forces_n, y_pred, label=f'Fit: y = {slope:.2f}x + {intercept:.2f}', color='red')
plt.xlabel('Force (N)')
plt.ylabel('ADC Value')
plt.title('Force Sensor Calibration')
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_dir, "calibration_curve.png"))
plt.show()

# === CLOSE SERIAL ===
ser.close()
