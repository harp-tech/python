# Migrating from harp-python

`harp-data` is the successor to `harp-python` for reading Harp binary data files into
pandas DataFrames. The core concepts of device schemas, register maps and binary files
are unchanged, but the API has been reorganized to separate data reading from device
communication.

This guide covers the three workflows most users relied on in `harp-python`.

## Swap the package

Replace the old dependency:

```sh title="Before"
pip install harp-python
```

```sh title="After"
pip install harp-data
```

For the full toolkit of serial transport, device client and data reading, install the
umbrella package instead:

```sh
pip install harp
```

## Loading a device schema at runtime

In `harp-python`, `harp.create_reader()` accepted a dataset folder and handled
schema loading internally. `open_dataset` is the direct replacement, and it finds the
`device.yml` inside the folder automatically:

```python title="Before"
import harp

reader = harp.create_reader("session.harp")
```

```python title="After"
from harp.data import open_dataset

reader = open_dataset("session.harp")
```

If the schema lives outside the data folder, pass it explicitly:

```python
reader = open_dataset("session.harp", schema="/path/to/device.yml")
```

### Finding out what a session holds

`reader.contents` maps the name of every register with data in the folder to its address,
in address order, which is the quickest way to see what was recorded:

```python
reader = open_dataset("session.harp")
print(reader.contents)   # {'WhoAmI': 0, 'DigitalInputState': 32, ...}
```

The reader also holds the compiled device module at `reader.device_module`, so a register
class can be reached without keeping a separate variable:

```python
df = reader.read(reader.device_module.AnalogData)
```

!!! note
    The old `harp.read_schema()` had no direct equivalent that needed calling separately.
    `open_dataset` handles schema loading in one step, matching the convenience
    of the original API.

## Reading a single register

The old API allowed reads through attribute access on the reader. The new
API inverts this, so `reader.read()` takes the register class, its name or
its address as the argument.

```python title="Before"
# by attribute name
df = reader.AnalogData.read()

# by name string
df = reader.registers["AnalogData"].read()

# by address
df = reader.registers[44].read()
```

```python title="After"
# by register class (accessed through the reader)
df = reader.read(reader.device_module.AnalogData)

# by name, the direct analogue of the old string lookup
df = reader.read("AnalogData")

# by address
df = reader.read(44)
```

Names resolve against the device address space rather than the module namespace, so a
common register such as `reader.read("WhoAmI")` works even though a device module does
not name it.

### Absolute timestamps

The `epoch` parameter keeps its name and moves from `create_reader` to `open_dataset`.
`harp-python` also allowed it per read. `harp-data` sets it once for the dataset, so
every register is read on the same clock.

```python title="Before"
reader = harp.create_reader("session.harp", epoch=harp.REFERENCE_EPOCH)
df = reader.AnalogData.read()
```

```python title="After"
from harp.data import REFERENCE_EPOCH

reader = open_dataset("session.harp", epoch=REFERENCE_EPOCH)
df = reader.read(reader.device_module.AnalogData)
```

### Reading the whole session at once

There is no `read_all()`. Whole-session loading is a comprehension over `contents`,
which keeps the choice of what to load with the caller:

```python
everything = {name: reader.read(name) for name in reader.contents}
```

### Bitmask registers lose their per-flag columns by default

This is the change most likely to break working code. `harp-python` always expanded a
bitmask register into one boolean column per flag, so a script could select a flag by
name. `harp-data` returns a single integer column instead, and expands the flags only
when asked:

```python title="Before"
led = reader.DigitalOutputSet.read()["GP15"]
```

```python title="After"
led = reader.read("DigitalOutputSet", demux_bit_masks=True)["GP15"]
```

Group masks need no such flag. `harp-python` mapped each value to its member name, and
`harp-data` decodes them by default, as a `pd.Categorical` rather than plain strings, so
a comparison against a string still reads naturally.

### Parameter reference

| harp-python | harp-data | Notes |
|---|---|---|
| `keep_type=True` | `keep_type=True` | Same, but the column is named `message_type` rather than `MessageType` |
| `epoch=REFERENCE_EPOCH` | `epoch=REFERENCE_EPOCH` | Same, but set on `open_dataset` rather than per read |
| `epoch=None` | `epoch=None` (default) | Float seconds, same |
| inferred from the first frame | `time_index=False` | Needed for data carrying no timestamp |
| always on | `demux_bit_masks=True` | Needed to keep per-flag columns |
| always on | `decode_enums=True` (default) | Group mask values, now `pd.Categorical` |

## Schemaless read

For a raw `.bin` file with no schema, or a quick look at the data, the `read()`
function works the same as before. Only the import path changes:

```python title="Before"
import harp

df = harp.read("Behavior_44.bin")
df = harp.read("Behavior_44.bin", keep_type=True)
```

```python title="After"
from harp.data import read

df = read("Behavior_44.bin")
df = read("Behavior_44.bin", keep_type=True)
```

Both functions infer the payload type and element count from the frame, so no register
metadata is needed. The new one assumes timestamped data, which is what a device sends,
and takes `time_index=False` for the rare buffer that is not. It also takes `epoch` per
call, since a single file has no dataset to set one on.

## Going further: static device packages

Loading a YAML at runtime is convenient, but for production workflows, or where IDE
autocompletion and type checking matter, a generated device package gives the same
interface without parsing a schema at startup.

Such a package is an ordinary Python module under the `harp.device` namespace. Import
it, pass it to `open_dataset`, and the rest of the API is identical:

```python
from harp.device import behavior
from harp.data import open_dataset

reader = open_dataset("session.harp", behavior)

# a register class now resolves statically
df = reader.read(behavior.AnalogData)
```

A generated module starts up faster and resolves under a type checker, which a module
built from a schema at runtime cannot. See
[Generating Registers from a Schema](../examples/create_device_module/create_device_module.md)
for how device modules are structured.
