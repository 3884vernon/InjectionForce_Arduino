import serial
import time
import csv
import matplotlib.pyplot as plt
from collections import deque

# === USER SETTINGS ===
PORT = '/dev/tty.usbmodem4827E2E15FC02'  # <-- your Arduino port
BAUD_RATE = 9600
CSV_FILENAME = 'force_readings_syringe_1cc_150_3.csv'
MAX_POINTS = 100
DURATION = 10  # seconds to run

# === SETUP SERIAL ===
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # allow time for connection to establish

# === SETUP CSV ===
csv_file = open(CSV_FILENAME, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Time (s)', 'ADC Value'])

# === SETUP PLOT ===
plt.ion()
fig, ax = plt.subplots()
x_data = deque(maxlen=MAX_POINTS)
y_data = deque(maxlen=MAX_POINTS)
line, = ax.plot([], [], 'b-', lw=2)
ax.set_ylim(0, 1023)
ax.set_xlim(0, 10)
ax.set_xlabel('Time (s)')
ax.set_ylabel('ADC Value')
ax.set_title('Real-Time FlexiForce Output')

start_time = time.time()

# === MAIN LOOP ===
try:
    while True:
        # Stop after 35 seconds
        elapsed_time = time.time() - start_time
        if elapsed_time > DURATION:
            print("35 seconds reached. Stopping data collection.")
            break

        raw = ser.readline().decode('utf-8', errors='ignore').strip()
        if "Force Signal =" in raw:
            adc_value = int(raw.split('=')[1].strip())
            timestamp = elapsed_time
            x_data.append(timestamp)
            y_data.append(adc_value)
            csv_writer.writerow([timestamp, adc_value])
            print(f"{timestamp:.2f}s: ADC = {adc_value}")

            # Update plot
            line.set_data(x_data, y_data)
            ax.set_xlim(max(0, timestamp - 10), timestamp + 1)
            ax.set_ylim(0, 1023)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    ser.close()
    csv_file.close()
    print(f"\nData saved to: {CSV_FILENAME}")
    plt.ioff()
    plt.show()
