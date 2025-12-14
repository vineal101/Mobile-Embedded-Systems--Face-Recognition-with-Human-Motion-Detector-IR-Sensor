import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

PIR_PIN = 25     # PIR OUT pin connected to GPIO25
GPIO.setup(PIR_PIN, GPIO.IN)

print("PIR Motion Sensor Ready...")
time.sleep(2)  # allow sensor to stabilize

try:
    while True:
        if GPIO.input(PIR_PIN):
            print("Motion Detected!")
        else:
            print("No Motion")

        time.sleep(0.5)

except KeyboardInterrupt:
    GPIO.cleanup()

