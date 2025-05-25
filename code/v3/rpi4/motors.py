import RPi.GPIO as GPIO
import time
import gyroscope

# Set GPIO mode
GPIO.setmode(GPIO.BOARD)

class Motors:
    def __init__(self):
        self.gyro = gyroscope.MPU6050()

        # Motor pins
        self.IN1 = 11
        self.IN2 = 13
        self.ENA = 15  # PWM for motor 1
        self.IN3 = 16
        self.IN4 = 18
        self.ENB = 22  # PWM for motor 2

        # Setup GPIO
        for pin in [self.IN1, self.IN2, self.ENA, self.IN3, self.IN4, self.ENB]:
            GPIO.setup(pin, GPIO.OUT)

        # Setup PWM at 1kHz
        self.pwmA = GPIO.PWM(self.ENA, 1000)
        self.pwmB = GPIO.PWM(self.ENB, 1000)

        # Start PWM at 0% duty
        self.pwmA.start(0)
        self.pwmB.start(0)

    def forward(self, speed):
        self.pwmA.ChangeDutyCycle(speed)
        self.pwmB.ChangeDutyCycle(speed)
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)

    def stop(self):
        self.pwmA.ChangeDutyCycle(0)
        self.pwmB.ChangeDutyCycle(0)

    def left90(self):
        self.pwmA.ChangeDutyCycle(0)
        self.pwmB.ChangeDutyCycle(100)
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)

        yaw_angle = 0
        last_time = time.time()

        while True:
            gyro_z = self.gyro.get_gyro_z()
            current_time = time.time()
            dt = current_time - last_time
            yaw_angle += gyro_z * dt
            last_time = current_time

            if abs(yaw_angle) >= 90:
                self.stop()
                break

    def right90(self):
        self.pwmA.ChangeDutyCycle(100)
        self.pwmB.ChangeDutyCycle(0)
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)

        yaw_angle = 0
        last_time = time.time()

        while True:
            gyro_z = self.gyro.get_gyro_z()
            current_time = time.time()
            dt = current_time - last_time
            yaw_angle += gyro_z * dt
            last_time = current_time

            if abs(yaw_angle) >= 90:
                self.stop()
                break

    def turn_around180(self):
        self.pwmA.ChangeDutyCycle(100)
        self.pwmB.ChangeDutyCycle(100)
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.HIGH)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)

        yaw_angle = 0
        last_time = time.time()

        while True:
            gyro_z = self.gyro.get_gyro_z()
            current_time = time.time()
            dt = current_time - last_time
            yaw_angle += gyro_z * dt
            last_time = current_time

            if abs(yaw_angle) >= 180:
                self.stop()
                break

    def run(self, width_frame, cx):
        middle = width_frame // 2
        if cx < middle - 50:
            print("drive left")
            self.left90(cx)
        elif cx > middle + 50:
            print("drive right")
            self.right90(cx)
        else:
            print("drive forward")
            self.forward(100)

    def cleanup(self):
        self.stop()
        GPIO.cleanup()
