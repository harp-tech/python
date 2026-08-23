from harp import serial
from harp.device import core

SERIAL_PORT = "/dev/ttyUSB0"  # or "COMx" in Windows, where "x" is the serial port number

# Omitting the device argument gives schema-free access, which skips the identity
# check, so this works against any device. The connection closes on exit.
with serial.open_device(port=SERIAL_PORT) as device:
    # Identify the device.
    print("WhoAmI:", device.read(core.WhoAmI).payload)

    # Dump every core register.
    for address, register in sorted(core.REGISTER_MAP.items()):
        reply = device.read(register)
        print(f"{register.__name__:24s} (addr {address:2d}) = {reply.payload}")
