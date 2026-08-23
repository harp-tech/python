# harp-protocol

[![PyPI version](https://badge.fury.io/py/harp-protocol.svg)](https://badge.fury.io/py/harp-protocol)

The Harp Protocol is a binary communication protocol created to facilitate and unify the interaction between different devices. It was designed with efficiency and ease of parsing in mind.

For more detail please check the [official Harp Tech documentation](https://harp-tech.org/protocol/BinaryProtocol-8bit.html).

`harp-protocol` provides the building blocks: message framing and the typed register/payload DSL. Each register builds its frames with `format` and decodes them with `parse`.

```python
import numpy as np
from harp.protocol import HarpMessage, RegisterU16

class WhoAmI(RegisterU16):
    address = 0

frame = WhoAmI.format(np.uint16(1216))           # build a Write frame
value = WhoAmI.parse(HarpMessage.parse(frame))   # -> np.uint16(1216)
```

## Register value types

`parse` returns numpy scalars rather than plain `int` or `float`, so a value carries the width its register declares. A Python `int` has no width and no upper bound, so it cannot distinguish a `U8` from a `U32`, nor detect a value leaving the register range.

```python
np.uint16(65535) + 1   # RuntimeWarning: overflow encountered in scalar add
65535 + 1              # 65536, wider than the register can hold
```

Numpy scalars behave like plain Python numbers in arithmetic, comparison, and formatting. Use `int()` or `float()` where a built-in type is required.

## Read an address no schema describes

A register class is normally declared with its address, as above, or generated from a `device.yml`. Calling a register base with an address instead builds a one-off register for that address, which is how a payload outside any schema is read and written:

```python
from harp.protocol import RegisterU8Array, RegisterU16

uid = RegisterU8Array(0x10, length=16)   # R_UID, named by no schema
tag = RegisterU8Array(0x11, length=16)   # R_TAG, the firmware git hash
version = RegisterU16(0x08)              # any address, as a scalar
```

The result is an ordinary register, so it goes through `read` and `write` on a device exactly as a declared one does. `length` is keyword-only for the array form, and it is the element count rather than a byte count. An already-addressed register rejects the call, so `WhoAmI(44)` raises rather than quietly producing a register at another address.

It carries no transport or device logic. See [`harp-device`](https://github.com/harp-tech/python/tree/main/src/packages/harp-device) for the device layer.

`harp-protocol` is released as open source under the [MIT license](https://github.com/harp-tech/python/blob/main/LICENSE). Bug reports and contributions are welcome at [the GitHub repository](https://github.com/harp-tech/python).
