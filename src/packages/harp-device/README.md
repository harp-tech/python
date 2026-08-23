# harp-device

The transport-agnostic device layer for the Harp protocol: the core Harp registers and a `Device` base that handles framing, request/reply and register access. It depends only on [`harp-protocol`](https://github.com/harp-tech/python/tree/main/src/packages/harp-protocol), with no transport dependencies. Pair it with a transport such as [`harp-serial`](https://github.com/harp-tech/python/tree/main/src/packages/harp-serial).

## Read/write registers

A `Device` operates over a transport. `read` and `write` take a register class:

```python
from harp.device import core

# `device` is a Device opened over some transport, see harp-serial
who = device.read(core.WhoAmI).payload          # -> np.uint16
device.write(core.OperationControl, payload)   # write a register
```

## When a request fails

An error reply raises `DeviceError`, which keeps the reply as `reply` so the frame sent by the device stays available for inspection. Pass `raise_on_error=False` to the constructor to receive such a reply as an ordinary return value instead. A transport failure raises `TransportError`, and every later request reports the same failure rather than waiting for a reply that cannot arrive. A device that never answers raises `TimeoutError` after `REPLY_TIMEOUT`, which is also what happens when `close` is called during a request.

## Extending for a specific device

A device is described by a module. Downstream, often generated, packages record the device identity as `WHO_AM_I`, declare the register classes at module level, and expand the core `REGISTER_MAP` beside them:

```python
from harp.device.core import REGISTER_MAP as _CORE_REGISTER_MAP

WHO_AM_I: int = 1216
REGISTER_MAP = {**_CORE_REGISTER_MAP, 32: DigitalInputState, ...}
```

This is the same structure `create_device_module` builds from a schema, so a device reads the same way whether it was generated ahead of time or compiled at runtime. A `WHO_AM_I` of `0` marks an unregistered device, used while a device is in development or outside the official registry, and identity checks are skipped for it.

A device module names only what its schema declares, the registers beside the enums and payload classes they are built from, so `REGISTER_MAP` is the device address space while the module namespace is what the device adds to it. The core registers and any core mask the schema reuses have a single definition, in `harp.device.core`, and are reached from there rather than through the device module. The core register set is not a device, so it carries no `WHO_AM_I`.

Pass the module to `Device`, or to `open_device`, to validate identity on open:

```python
from harp.device import behavior, client, core

with client.Device(transport, behavior) as device:
    device.read(core.WhoAmI)                 # a core register
    device.read(behavior.DigitalInputState)  # declared by the schema
```

The `WHO_AM_I` in the module determines the check, and `0` skips it. Omitting the module skips validation. The module is not otherwise consulted: registers reach `read`, `write` and `subscribe` as arguments either way, and only a subscribed register is parsed on arrival. Core registers such as `WhoAmI` and `OperationControl` come from `harp.device.core` and are read the same way.

A new transport is just an object implementing the `ITransport` protocol, with `open`, `write`, `read` and `close`.

## Generating registers from a `device.yml`

Without a pre-generated device package, `create_device_module` builds the same structure at runtime from Harp `device.yml` text: register, enum and payload classes at module level, a `REGISTER_MAP` beside them, and the identity declared by the schema as `WHO_AM_I`. Identifiers match a generated package name for name: register, enum, and payload class names come from the yml verbatim, payload fields are `snake_case`, and enum members are `SCREAMING_SNAKE_CASE`. A `maskType` the schema does not declare resolves against the core masks, and a register marked `private` is emitted with an underscore-prefixed name.

```python
from pathlib import Path

from harp.device import schema

behavior = schema.create_device_module(Path("device.yml").read_bytes())
reg = behavior.AnalogData          # by name
reg = behavior.REGISTER_MAP[44]    # or by address
```

The module is not registered in `sys.modules`, so it has to be bound rather than imported. Names come from the schema at runtime, so they don't autocomplete and aren't statically checked. A generated package on disk gives both.

For a custom `interfaceType`, pass its converter via `converters=`, keyed by `{InterfaceType}Converter` or `{MemberName}Converter`. An unresolved custom type raises `UnknownConverterError`, or pass `require_converters=False` to decode it natively:

```python
schema.create_device_module(yml_text, converters={"DataConverter": DataConverter()})
```

`parse_device_schema(yml_text)` is also public, returning the parsed schema model without a module: registers, masks, and optional device identity.

`harp-device` is released as open source under the [MIT license](https://github.com/harp-tech/python/blob/main/LICENSE). Bug reports and contributions are welcome at [the GitHub repository](https://github.com/harp-tech/python).
