# Registers from a schema

This example demonstrates how to turn a Harp `device.yml` into a module of register classes at runtime with `create_device_module`, without a code-generation step. This is the quickest way to get started given only the schema of a device and no pre-generated package for it.

A generated device package is a module: register classes at module level, with a `REGISTER_MAP` beside them keyed by address. `create_device_module` builds that same structure from a schema, so registers are accessed the same way, either by name as `behavior.AnalogData` or by address as `behavior.REGISTER_MAP[44]`. From there they work exactly like the registers of a pre-generated package. Pass the module to [`Device`](../api/device.md) to talk to hardware, which validates the device identity on open, or use the registers with [`parse_to_dataframe`](../api/data.md) to decode recorded data.

For when an installed device package is the better choice, see [Choose a device module](../articles/device-modules.md).

{% include-markdown "includes/serial-port.md" %}

<!--codeinclude-->
```python
[](./registers_from_schema.py)
```
<!--/codeinclude-->
