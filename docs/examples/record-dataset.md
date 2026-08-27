# Record a dataset folder

`harp.device.client.attach_writer` records everything a device emits into a **de-multiplexed dataset folder**, the layout [Read a dataset folder](dataset.md) reads back: one binary file per register, named `<DeviceName>_<address>.bin`, next to a copy of the `device.yml` the module was built from.

Each message is appended to the file of its own register as the complete Harp frame it arrived as, header and checksum included, which is what the reference C# writer records. The file of a register is created the first time a message for it arrives, so the folder holds exactly the registers that were seen.

The writer owns its subscription, so leaving the `with` block both detaches it from the device and closes the files. All three message types are recorded by default: the register dump a device performs on request arrives as `Read` messages, so recording only `Event` would drop the configuration the session ran under.

{% include-markdown "includes/serial-port.md" %}

<!--codeinclude-->
```python
[](./record_dataset.py)
```
<!--/codeinclude-->
