# harp-data

Load Harp register data into pandas DataFrames. This is the package that pulls in `pandas`. [`harp-protocol`](https://github.com/harp-tech/python/tree/main/src/packages/harp-protocol) stays numpy-only and exposes a pandas-free `ColumnData` view that this package assembles into a DataFrame.

There are two ways in, depending on what is on disk:

- a whole **dataset folder** holding many registers, read with `DatasetReader`
- a single **register file** or buffer, read with `parse_to_dataframe`

## Read a whole dataset folder

A Harp acquisition is usually saved as a de-multiplexed folder, one binary file per register, named `<DeviceName>_<address>.bin`, alongside the `device.yml` schema for the device:

```text
📦 session.harp
 ┣ 📜 Behavior_0.bin
 ┣ 📜 Behavior_44.bin
 ┣ ...
 ┗ 📜 device.yml
```

Reading is based on a [device module](https://github.com/harp-tech/python/tree/main/src/packages/harp-device) that describes how to decode each register. `open_dataset` supplies one automatically. It finds the `device.yml` in the folder, builds the module, and returns a ready-to-use reader:

```python
from harp import data

reader = data.open_dataset("session.harp")
df = reader.read("AnalogData")  # by name
df = reader.read(44)            # by address
```

`contents` maps every register with data in the folder to its address, keyed by register name. It is the place to start on an unfamiliar dataset, and since its keys are exactly what `read` takes, loading a whole dataset can be done with a comprehension:

```python
reader.contents  # {'WhoAmI': 0, 'AnalogData': 33, ...}

frames = {name: reader.read(name) for name in reader.contents}
```

A name is resolved through the device register map rather than the module namespace, so the common registers are reachable by name too.

A register declared in the device register map with no data present in the folder reads as an empty DataFrame carrying the same columns, since the schema describes the structure of the data regardless of whether anything was recorded. `contents` is what tells the two cases apart. A register the device does not declare at all raises `KeyError`.

Given a device module already in hand, either a pre-generated package or one built with `create_device_module`, pass it as the second argument:

```python
from harp import data
from harp.device import behavior

reader = data.open_dataset("session.harp", behavior)
df = reader.read(behavior.AnalogData)  # by register class
```

Prefer the register class where a generated package supplies one, since it is the only form that type-checks and a misspelling is caught before the folder is read. A module built by `create_device_module` resolves its registers as `Any`, so there the class verifies no more than the name does.

The Harp time becomes the DataFrame index named `"Time"`, as float seconds by default or an absolute `DatetimeIndex` when the dataset is opened with `epoch=REFERENCE_EPOCH`. The anchor is set once for the dataset, since it describes how the recording was made rather than how one register is read. Data carrying no timestamp raise unless `time_index=False` is passed. Multi-chunk registers logged as `<DeviceName>_<address>_<suffix>.bin` are concatenated in filename order; pass a `resolver` to support an alternative on-disk layout. `paths` reports what the resolver found, keyed by address, which is where a custom layout or a chunked register can be checked.

The `<DeviceName>` prefix comes from the `DEVICE_NAME` declared by the device module. Pass `name=` to override it, or to supply one when the module declares an empty name.

When a device module declaring an identity is supplied and the folder carries a `device.yml`, their `whoAmI` values are checked against each other. Reusing a module across sessions and reaching the wrong folder then fails on construction rather than decoding the files against the wrong register map. Pass `validate=False` to turn off every check the reader performs, so a folder whose `device.yml` is damaged can be read with a module obtained elsewhere.

## Read a single register file

`parse_to_dataframe` takes a register and a source, either a path, bytes, or an open binary file, and returns one row per frame:

```python
from harp import data
from my_device import AnalogData

df = data.parse_to_dataframe(AnalogData, "AnalogData.bin")
df = data.parse_to_dataframe(
    AnalogData, raw, time_index=True, epoch=None, keep_type=False, decode_enums=True
)
```

`time_index` decides the index: `True`, the default, gives the Harp time named `"Time"`, and `False` gives a `RangeIndex`. `epoch` anchors that index, giving float seconds when omitted and an absolute `DatetimeIndex` when set to a datetime such as `REFERENCE_EPOCH`. This function reads one file rather than a dataset, so it takes the anchor directly. Enum fields decode to `pd.Categorical`, and `decode_enums=False` keeps raw codes.

## From an already-parsed payload

Given a batched payload already in hand, for example from `register.parse_bulk`, convert it directly:

```python
from harp import data

_data, timestamps, _msg, payload = AnalogData.parse_bulk(raw)
df = data.payload_to_dataframe(payload)
```

## Write data back out

`to_file` and `to_buffer` are the inverse of the readers, encoding values as Harp frames. Useful for round-tripping data or generating test corpora:

```python
from harp import data

data.to_file(AnalogData, values, "AnalogData.bin", timestamps=seconds)
```

`harp-data` is released as open source under the [MIT license](https://github.com/harp-tech/python/blob/main/LICENSE). Bug reports and contributions are welcome at [the GitHub repository](https://github.com/harp-tech/python).
