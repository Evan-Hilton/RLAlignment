from src.platform_control.stewart import StewartPlatform
import time
from serial.tools import list_ports
import numpy as np

# use this code to see the device list, then copy and paste the correct one into the port variable
# for port in list_ports.comports():
#     print(port.device, "-", port.description)

# initialize platform
port = "/dev/cu.usbmodemE8F60ABF50482"
platform =  StewartPlatform(port)
platform.send_command("L") # turns on the laser
platform.send_command("MAXA 1") # sets the rotation scale to 1 deg
platform.send_command("MAXP 1") # sets the position scale to 1 mm

_ = input()

platform.send_motion_command("Z 10")
platform.send_motion_command("Z -20")
platform.send_motion_command("Z 10")

_ = input()

platform.send_motion_command("X 10")
platform.send_motion_command("X -20")
platform.send_motion_command("X 10")

_ = input()

platform.send_motion_command("Y 10")
platform.send_motion_command("Y -20")
platform.send_motion_command("Y 10")

_ = input()

platform.send_motion_command("RZ 10")
platform.send_motion_command("RZ -20")
platform.send_motion_command("RZ 10")

_ = input()

platform.send_motion_command("RX 10")
platform.send_motion_command("RX -20")
platform.send_motion_command("RX 10")

_ = input()

platform.send_motion_command("RY 10")
platform.send_motion_command("RY -20")
platform.send_motion_command("RY 10")

_ = input()

platform.close()