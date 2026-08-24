# harp-serial

Serial transport for [`harp-device`](https://github.com/harp-tech/python/tree/main/src/packages/harp-device). Provides `SerialTransport` and the `open_device` factory, which pairs a device module or a `Device` class with a serial port. This is the package that pulls in `pyserial`.

## Usage

Like the builtin `open`, the returned device is connected and ready. Use it in a `with` block for guaranteed cleanup:

```python
from harp import serial
from harp.device import behavior, core

# Use "COMx" on Windows, "/dev/ttyUSBx" on Linux.
with serial.open_device(behavior, port="COM3") as device:
    print(device.read(core.WhoAmI).payload)         # a common register
    print(device.read(behavior.AnalogData).payload) # a device register
```

Passing a device module validates the device identity on open. Pass a `Device` subclass instead to preserve its own type, or omit the argument entirely for schema-free access, which skips the identity check.
