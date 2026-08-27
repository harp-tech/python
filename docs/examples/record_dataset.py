from pathlib import Path

from harp import data
from harp import serial
from harp.device import client, core, schema

SERIAL_PORT = "/dev/ttyUSB0"  # or "COMx" in Windows, where "x" is the serial port number

behavior = schema.create_device_module(Path("device.yml").read_bytes())

with serial.open_device(behavior, port=SERIAL_PORT) as device:
    # The writer owns its subscription, so leaving this block detaches it from the
    # device and closes the files. `device.yml` is copied into the folder as it opens.
    with client.attach_writer(device, "session.harp") as writer:
        # Ask the device to report every register, so the folder records the
        # configuration the session ran under and not only its events. Do this once the
        # writer is attached, or the reply burst is missed.
        device.write(
            core.OperationControl,
            core.OperationControlPayload(
                operation_mode=core.OperationMode.ACTIVE,
                dump_registers=True,
                heartbeat=core.EnableFlag.ENABLED,
                mute_replies=False,
                operation_led=core.EnableFlag.ENABLED,
                visual_indicators=core.EnableFlag.ENABLED,
            ),
        )

        input("Recording. Press Enter to stop.\n")
        print("recorded:", {address: path.name for address, path in writer.paths.items()})

# The folder carries its own schema, so it reads back without a module in hand.
reader = data.open_dataset("session.harp")
print(reader.contents)
