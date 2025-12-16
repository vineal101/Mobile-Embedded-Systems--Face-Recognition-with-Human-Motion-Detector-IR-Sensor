import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

# Pins (BCM)
TRIG = 23
ECHO = 24
PIR_PIN = 18

# Config
THRESHOLD_CM = 35.0
SAMPLE_INTERVAL = 0.2
RUN_DURATION = 20  # seconds

print("Ultrasonic + PIR (RAW) starting...")

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(PIR_PIN, GPIO.IN)

GPIO.output(TRIG, False)
print("Waiting for sensors to settle...")
time.sleep(2)

start_time = time.time()

try:
    while time.time() - start_time < RUN_DURATION:

        # ---- Ultrasonic distance ----
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()

        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = round(pulse_duration * 17150, 2)  # cm

        ultrasonic_close = (distance <= THRESHOLD_CM)

        # ---- PIR motion ----
        motion_detected = (GPIO.input(PIR_PIN) == 1)

        # ---- Combined gate (for face detection later) ----
        person_close = ultrasonic_close and motion_detected

        print(
            f"Distance: {distance:6.2f} cm | "
            f"UltrasonicClose={int(ultrasonic_close)} | "
            f"PIR_Motion={int(motion_detected)} | "
            f"PERSON_CLOSE={int(person_close)}"
        )

        time.sleep(SAMPLE_INTERVAL)

finally:
    GPIO.cleanup()
    print("Finished.")

