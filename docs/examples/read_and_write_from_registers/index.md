# Read and Write from Registers

This example demonstrates how to read and write from registers, using the core registers exposed by `harp.device.core`. Device-specific registers, for example the digital I/O of a Harp Behavior device, are used the same way. Pass the register classes of that device to `read` and `write`.

{% include-markdown "includes/serial-port.md" %}

<!--codeinclude-->
```python
[](./read_and_write_from_registers.py)
```
<!--/codeinclude-->
