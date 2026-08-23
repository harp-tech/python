from pathlib import Path

from harp import data
from harp import serial
from harp.device import schema

SERIAL_PORT = "/dev/ttyUSB0"  # or "COMx" in Windows, where "x" is the serial port number

# `create_device_module` compiles a Harp `device.yml` into a module of register classes at
# runtime, with no code-generation step. This is the quickest way to work with a device
# that has no pre-generated package: point it at the schema and it produces the same
# structure a generated package has, registers at module level beside a
# `REGISTER_MAP`.
behavior = schema.create_device_module(Path("device.yml").read_bytes())

print("WhoAmI:", behavior.WHO_AM_I)  # device identity, taken from the schema
AnalogData = behavior.AnalogData  # registers are reached by name
assert behavior.REGISTER_MAP[44] is AnalogData  # or by address

# Registers are ordinary register classes, so they work with `read` and `write` on
# any `Device` over a transport. Passing the module itself validates the device
# identity on open, against its `WHO_AM_I`, which a value of `0` skips.
with serial.open_device(behavior, port=SERIAL_PORT) as device:
    print("AnalogData:", device.read(AnalogData).payload)

# The same register classes also decode a recorded register log file into a pandas
# DataFrame. See the "Reading a single register file" example for more.
df = data.parse_to_dataframe(AnalogData, "Behavior_44.bin")
print(df.head())


# --- Custom interface types --------------------------------------------------
# A register with a custom `interfaceType` needs a converter so its field decodes
# to the right Python type. Pass it via `converters=`, keyed by "<Name>Converter":
#
#   behavior = schema.create_device_module(yml_text, converters={"DataConverter": DataConverter()})
#
# An unresolved custom type raises `UnknownConverterError`. Pass `require_converters=False` to
# decode it natively instead. A register marked `private` in the schema is emitted
# with an underscore-prefixed name. For the parsed schema model rather than a module,
# `parse_device_schema(yml_text)` returns that directly.
