# Reading a dataset folder

A Harp acquisition is usually saved as a **de-multiplexed dataset folder**: one binary file per register, named `<DeviceName>_<address>.bin`, next to the `device.yml` schema for the device. `harp.data.DatasetReader` reads that whole folder into pandas DataFrames, based on a [device module](../api/device.md) that describes how to decode each register.

This is the recommended entry point for a recorded session on disk. To decode a single loose `.bin` file instead, see [Reading a single register file](register-file.md).

The quickest way in is `open_dataset(folder)`. It finds the `device.yml` inside the folder, builds the device module, and returns a reader ready to go. Given a device module already in hand, for example from a pre-generated package, pass it as the second argument, `open_dataset(folder, module)`. A register is then read by class, by name, or by address. The Harp time becomes the `"Time"` index, as float seconds or an absolute `DatetimeIndex` when the dataset is opened with an `epoch`.

<!--codeinclude-->
```python
[](./dataset.py)
```
<!--/codeinclude-->
