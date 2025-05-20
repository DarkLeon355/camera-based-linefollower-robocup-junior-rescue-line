"""
https://circuitofthings.com/raspberry-pi-motor-control-with-l298n/

L298N Motor Driver Logic:

IN1 & IN2 = 0 -> Motor off
IN1 = 1 & IN2 = 0 -> Motor forward
IN1 = 0 & IN2 = 1 -> Motor backward
IN1 & IN2 = 1 -> Motor off

-> Same for IN3 & IN4
"""

import RPi.GPIO as GPIO
import time

# Set the GPIO mode
GPIO.setmode(GPIO.BOARD)

class Motors:
    def __init__(self):
        self.IN1 = 11
        self.IN2 = 13
        self.ENA = 15 # PWM for motor 1
        self.IN3 = 16
        self.IN4 = 18
        self.ENB = 22 # PWM for motor 2

        #Left motor
        GPIO.setup(self.ENA, GPIO.OUT)
        GPIO.setup(self.IN1, GPIO.OUT)
        GPIO.setup(self.IN2, GPIO.OUT)

        #Right motor
        GPIO.setup(self.ENB, GPIO.OUT)
        GPIO.setup(self.IN3, GPIO.OUT)
        GPIO.setup(self.IN4, GPIO.OUT)

        self.pwmA = GPIO.PWM(self.ENA, 1000) # 1000Hz
        self.pwmB = GPIO.PWM(self.ENB, 1000) # 1000Hz

    def forward(self, speed): #Note, the speed is the duty cycle (0-100)
        self.pwmA.start(speed)
        self.pwmB.start(speed)
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)

    def left(self, cx): #this gets the average x coordinate of the line
        self.pwmA.start(0)
        self.pwmB.start(100)
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
            
    def right(self, cx): #this gets the average x coordinate of the line
        self.pwmA.start(100)
        self.pwmB.start(0)
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)


    def turn_around(self):
        self.pwmA.start(100)
        self.pwmB.start(100)
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.HIGH)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        time.sleep(0.5) #measure the time it takes to turn around

    def stop(self):
        self.pwmA.start(0)
        self.pwmB.start(0)
    
    def run(self, width_frame, cx):
        middle = width_frame // 2
        if cx < middle - 50:
            print("drive left")
            self.left(cx)
        elif cx > middle + 50:
            self.right(cx)
            print("drive right")
        else:
            self.forward(100)
            print("drive forward")
