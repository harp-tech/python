# Choosing a device module

A **device module** is what describes a Harp device to the library. It holds the register classes for that device at module level, a `REGISTER_MAP` keyed by address, and the payload and enum classes referenced by those registers. It is what `Device` needs to talk to hardware and what [`open_dataset`](../api/data.md) needs to read a recorded session.

A device module comes from one of two places. A generated device package is a real Python module on disk, produced from a `device.yml` by the Harp code generators and installed as a dependency. A runtime module is built in memory by [`create_device_module`](../api/device.md), from a `device.yml` read at the moment it is needed.

Both are the same kind of thing. Registers are reached the same way in either case, by name as `behavior.AnalogData` or by address as `behavior.REGISTER_MAP[44]`, and the names agree because both are derived from the same schema under the same naming convention. Analysis code written against one lines up name for name against the other.

The choice is therefore not about what the registers can do. It is about where the definitions come from, and what that costs.

## Benefits of a generated package

**Static typing and autocomplete.** The module exists on disk before the program runs, so an editor offers its register names and a type checker verifies them. A runtime module is built while the program runs, so neither can see it, and it is not in `sys.modules`, so it has to be bound to a name rather than imported.

**A version that can be pinned.** A generated package is an ordinary dependency, so it can be pinned in a lock file and every install resolves the same register definitions. A runtime module is only as stable as its `device.yml`, so the same analysis code can see different field names once that file changes.

**Converters for its own custom types.** A generated package includes the converters for any custom `interfaceType` declared by its schema. A runtime module has to be given them through `converters=`.

## Benefits of a runtime module

**No build step.** A `device.yml`, including one read straight off a device, becomes a working module in a single call. There is nothing to generate, install, or keep in step with the schema.

**Coverage for any device.** No published package is needed, so unreleased, custom and one-off schemas work immediately. It is also what makes a recorded session readable when the only description of the device is the `device.yml` saved beside it.

## Which to use

For a widely-used device with a published package, use the package. Better editor support, static typing and a version that can be pinned are worth a dependency for code that will be maintained.

Use `create_device_module` when no package exists, when the schema is still moving, or when a recorded session has to be read with nothing but the `device.yml` saved alongside it. See [Registers from a schema](../examples/registers-from-schema.md) for a worked example.
