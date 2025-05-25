import smbus
import time


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


sensor = MPU6050()
yaw_angle = 0
last_time = time.time()


# Example for turning 90 degrees
"""
while turning:
    gyro_z = sensor.get_gyro_z()        # °/s
    current_time = time.time()
    dt = current_time - last_time       # seconds

    yaw_angle += gyro_z * dt            # degrees
    last_time = current_time

    if abs(yaw_angle) >= 90:
        print("Stop turn")
"""