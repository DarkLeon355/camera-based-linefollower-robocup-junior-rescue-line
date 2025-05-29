import smbus2 as smbus
import time
import numpy as np


class MPU6050:
    def __init__(self):
        # I2C address and register definitions
        self.Device_Address = 0x68
        self.PWR_MGMT_1   = 0x6B
        self.SMPLRT_DIV   = 0x19
        self.CONFIG       = 0x1A
        self.GYRO_CONFIG  = 0x1B
        self.INT_ENABLE   = 0x38
        self.GYRO_ZOUT_H  = 0x47

        self.bus = smbus.SMBus(1)  # Use I2C bus 1 on Pi 4
        self.MPU_Init()
        self.GYRO_THRESHOLD = 5
        self.yaw_angle = 0  # Track total angle over time

    def MPU_Init(self):
        self.bus.write_byte_data(self.Device_Address, self.SMPLRT_DIV, 7)
        self.bus.write_byte_data(self.Device_Address, self.PWR_MGMT_1, 1)
        self.bus.write_byte_data(self.Device_Address, self.CONFIG, 0)
        self.bus.write_byte_data(self.Device_Address, self.GYRO_CONFIG, 24)  # ±2000°/s
        self.bus.write_byte_data(self.Device_Address, self.INT_ENABLE, 1)

    def read_raw_data(self, addr):
        high = self.bus.read_byte_data(self.Device_Address, addr)
        low = self.bus.read_byte_data(self.Device_Address, addr + 1)
        value = (high << 8) | low
        if value > 32767:
            value -= 65536
        return value

    def get_gyro_z(self):
        raw_z = self.read_raw_data(self.GYRO_ZOUT_H)
        return raw_z / 16.4  # Convert to degrees/sec (±2000°/s sensitivity)

    def update_angle(self, previous_time):
        gyro_z = self.get_gyro_z()
        self.current_time = time.time()
        dt = self.current_time - previous_time

        # Only add to yaw angle if rotation is above threshold
        if abs(gyro_z) > self.GYRO_THRESHOLD:
            self.yaw_angle += gyro_z * dt
        
        return self.yaw_angle


# Main program
sensor = MPU6050()
previous_time = time.time()

try:
    while True:
        current_angle = sensor.update_angle(previous_time)
        print(f"Yaw Angle: {current_angle:.2f}°")
        previous_time = sensor.current_time
        time.sleep(0.01)  # Small delay to prevent CPU overuse
except KeyboardInterrupt:
    print("Program stopped by user")


