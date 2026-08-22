# Generating Registers from a Schema

This example demonstrates how to turn a Harp `device.yml` into a module of register classes at runtime with `create_device_module`, without a code-generation step. This is the quickest way to get started given only the schema of a device and no pre-generated package for it.

A generated device package is a module: register classes at module level, with a `REGISTER_MAP` beside them keyed by address. `create_device_module` builds that same structure from a schema, so registers are reached the same way, either by name as `behavior.AnalogData` or by address as `behavior.REGISTER_MAP[44]`. From there they work exactly like the registers of a pre-generated package. Pass the module to [`Device`](../../api/device.md) to talk to hardware, which validates the device identity on open, or use the registers with [`parse_to_dataframe`](../../api/data.md) to decode recorded data.

## When to use runtime generation

`create_device_module` trades statically generated device packages for schema-driven convenience. Both sides of that trade-off are worth understanding.

**Benefits:**

- **No build step.** A `device.yml`, even one just pulled off a device, becomes a working module in a single call. There is nothing to generate, install, or keep in sync with the schema.
- **Coverage for any device.** No published package is needed. Unreleased, custom, or one-off schemas work immediately.
- **Names match the generated package.** Registers, fields, and enums come straight from the `device.yml`, under the same naming convention a generated package uses, so code written against either lines up name for name.

**Limitations:**

- **Static typing and autocomplete.** The names exist only once the module is built, so an editor cannot offer them and a type checker cannot verify them. A generated package is a real module on disk, so both work. The module is also not in `sys.modules`, so it has to be bound rather than imported.
- **Reproducibility.** A generated package is a versioned dependency, so it can be pinned in a lock file and every install resolves the same register definitions. A runtime module is built from the `device.yml`, so the same analysis code can see different field names when it changes.
- **Turn-key custom types.** A custom `interfaceType` must be injected via `converters=`, shown below, whereas a generated package ships its own converters.

For widely-used devices a pre-generated package remains the authoritative choice, with better editor support, static typing, and a pinnable version. Reach for `create_device_module` to go from a schema to working code with no code-generation step.

{% include-markdown "includes/serial-port.md" %}

<!--codeinclude-->
```python
[](./create_device_module.py)
```
<!--/codeinclude-->
