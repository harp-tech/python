# Migrating from harp-python

`harp-data` is the successor to `harp-python` for reading Harp binary data files into
pandas DataFrames. The core concepts are unchanged — device schemas, register maps,
binary files — but the API has been reorganized to separate data reading from device
communication.

This guide covers the three workflows most users relied on in `harp-python`.

---

## Swap the package

Replace the old dependency:

```sh title="Before"
pip install harp-python
```

```sh title="After"
pip install harp-data
```

If you want the full toolkit — serial transport, device client, and data reading — install
the umbrella package instead:

```sh
pip install harp
```

---

## Loading a device schema at runtime

In `harp-python`, `harp.create_reader()` accepted a dataset folder and handled
schema loading internally. `create_dataset_reader` is the direct replacement — it
finds the `device.yml` inside the folder automatically:

```python title="Before"
import harp

reader = harp.create_reader("session.harp")
```

```python title="After"
from harp.data import create_dataset_reader

reader = create_dataset_reader("session.harp")
```

If the schema lives outside the data folder, pass it explicitly:

```python
reader = create_dataset_reader("session.harp", schema="/path/to/device.yml")
```

### Accessing the device module

The reader holds a reference to the compiled device module at `reader.device_module`.
Use it to look up register classes by name — no need to keep a separate variable:

```python
reader = create_dataset_reader("session.harp")

# access any register class through the reader
df = reader.read(reader.device_module.AnalogData)
```

!!! note
    The old `harp.read_schema()` had no direct equivalent you needed to call separately.
    `create_dataset_reader` handles schema loading in one step, matching the convenience
    of the original API.

---

## Reading a single register

The old API gave you attribute access on the reader — `reader.AnalogData.read()`. The
new API inverts this: you call `reader.read()` and pass the register class or its
address as the argument.

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

# by address
df = reader.read(44)
```

### Absolute timestamps

The `epoch` parameter moves from the reader constructor into the `read()` call:

```python title="Before"
reader = harp.create_reader("session.harp", epoch=harp.REFERENCE_EPOCH)
df = reader.AnalogData.read()
```

```python title="After"
from harp.data import REFERENCE_EPOCH

df = reader.read(reader.device_module.AnalogData, epoch=REFERENCE_EPOCH)
```

### Reading the whole session at once

`read_all()` returns a dictionary of DataFrames keyed by register name. Registers
with no corresponding `.bin` file are skipped:

```python
everything: dict[str, pd.DataFrame] = reader.read_all()
```

### Parameter reference

| harp-python | harp-data | Notes |
|---|---|---|
| `keep_type=True` | `message_type=True` | Renamed |
| `epoch=REFERENCE_EPOCH` | `epoch=REFERENCE_EPOCH` | Same |
| `epoch=None` | `epoch=None` (default) | Float seconds; same |
| — | `decode_enums=True` | New: enum fields as `pd.Categorical` |
| — | `demux_bit_masks=False` | New: expand bitmask flags into one column per flag |

---

## Schemaless read

If you have a raw `.bin` file and no schema — or you just want to inspect the data
quickly — the `read()` function works the same as before. Only the import path and
one parameter name change:

```python title="Before"
import harp

df = harp.read("Behavior_44.bin")
df = harp.read("Behavior_44.bin", keep_type=True)
```

```python title="After"
from harp.data import read

df = read("Behavior_44.bin")
df = read("Behavior_44.bin", message_type=True)
```

Both functions infer the payload type and element count automatically from the first
frame — no register metadata needed.

---

## Going further: static device packages

Loading a YAML at runtime is convenient, but for production workflows — or when you
want IDE autocompletion and type safety — Harp device packages are pre-compiled Python
modules that give you the same interface without any schema parsing at startup.

A static device package installs its register map as a proper Python module. You
import it, pass it directly to `DatasetReader`, and the rest of the API is identical:

```python
pip install harp-device-behavior
```

```python
from harp.device.behavior import device as behavior
from harp.data import DatasetReader

reader = DatasetReader(behavior, "session.harp")

# everything works the same
df = reader.read(behavior.AnalogData)
everything = reader.read_all()
```

The static module is faster to start up and ships with stubs for autocompletion. See
[Generating Registers from a Schema](../examples/create_device_module/create_device_module.md)
for how device modules are structured.
