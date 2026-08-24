import numpy as np

from harp import serial
from harp.device import client, core
from harp.protocol import HarpMessage

SERIAL_PORT = "/dev/ttyUSB0"  # or "COMx" in Windows, where "x" is the serial port number


def print_timestamp(msg: HarpMessage[np.uint32]) -> None:
    print(f"[timestamp] {msg.timestamp:.6f}  {msg.payload}")


def print_any_event(msg: HarpMessage) -> None:
    register = core.REGISTER_MAP.get(msg.address, None)
    value = register.parse(msg) if register is not None else msg.payload_bytes.hex()
    print(f"[{msg.address}] {msg.timestamp:.6f}  {msg.message_type.name:<5s}  {value}")


with serial.open_device(client.Device, port=SERIAL_PORT) as device:
    # Subscribe to a single register: the handler receives a message typed by its payload.
    timestamp_subscription = device.subscribe(core.TimestampSeconds, print_timestamp)

    # Subscribe to every register at once: the handler receives the raw message.
    device.subscribe_all(print_any_event)

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

    input("Listening for events. Press Enter to stop.\n")
    timestamp_subscription.unsubscribe()
