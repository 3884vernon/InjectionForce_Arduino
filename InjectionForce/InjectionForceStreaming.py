import serial
import time

ser = serial.Serial('/dev/tty.usbmodem4827E2E15FC02', 9600)
time.sleep(2)

print("Hi THere! :)")

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        print(f"RECEIVED: {line}")
except KeyboardInterrupt:
    ser.close()
