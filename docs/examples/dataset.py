from harp import data
from harp.device import core

# A Harp acquisition is usually saved as a de-multiplexed dataset folder, one
# `.bin` file per register, named "<DeviceName>_<address>.bin", next to the
# `device.yml` schema for the device:
#
#   📦 session.harp
#    ┣ 📜 Behavior_0.bin
#    ┣ 📜 Behavior_44.bin
#    ┣ ...
#    ┗ 📜 device.yml
#
# `open_dataset` does the right thing: it finds `device.yml` inside the folder,
# builds the module of register classes that knows how to decode each register,
# and hands back a reader ready to go.
reader = data.open_dataset("session.harp")

# Read one register into a DataFrame by register class, which covers any register
# in the device map, including core ones such as `OperationControl`.
df = reader.read(core.OperationControl)

# A register can also be read by name. Names resolve through the device register
# map rather than the module namespace, so core registers are accessible too.
df = reader.read("OperationControl")

# Or by address. The Harp time becomes the DataFrame index, named "Time",
# holding float seconds from device start.
df = reader.read(44)
print(df.head())

# `contents` names every register that has data in this folder. Most datasets log
# every register by default.
frames = {name: reader.read(name) for name in reader.contents}
print(list(frames))

# Opening with an epoch turns the "Time" index into an absolute `DatetimeIndex`
# instead of float seconds, for every register of the dataset. `REFERENCE_EPOCH` is
# time zero of the Harp clock in UTC.
absolute = data.open_dataset("session.harp", epoch=data.REFERENCE_EPOCH).read(44)
print(absolute.index[:3])

# --- Working from a device module already in hand ----------------------------
# A pre-generated device package, or one built with `create_device_module`, is
# passed as the second argument. Either way the device identity is checked against
# the `device.yml` in the folder, so a module paired with the wrong session fails
# here rather than decoding against the wrong register map. A generated package
# adds register classes a type checker can verify:
#
#   from pathlib import Path
#
#   from harp import data
#   from harp.device import schema
#
#   behavior = schema.create_device_module((Path("session.harp") / "device.yml").read_bytes())
#   reader = data.open_dataset("session.harp", behavior)
#   df = reader.read(behavior.AnalogData)
