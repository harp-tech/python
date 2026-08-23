from harp import serial
from harp.device import client, core

SERIAL_PORT = "/dev/ttyUSB0"  # or "COMx" in Windows, where "x" is the serial port number

with serial.open_device(client.Device, port=SERIAL_PORT) as device:
    # Read a scalar register.
    print("WhoAmI:", device.read(core.WhoAmI).payload)

    # Read a structured register and inspect a field.
    control = device.read(core.OperationControl).payload
    print("operation_mode before:", control.operation_mode)

    # Write the register, then read it back to confirm the change. A struct payload
    # is built whole, so every field is given a value.
    device.write(
        core.OperationControl,
        core.OperationControlPayload(
            operation_mode=core.OperationMode.ACTIVE,
            dump_registers=False,
            mute_replies=False,
            visual_indicators=core.EnableFlag.ENABLED,
            operation_led=core.EnableFlag.ENABLED,
            heartbeat=core.EnableFlag.DISABLED,
        ),
    )
    control = device.read(core.OperationControl).payload
    print("operation_mode after:", control.operation_mode)
