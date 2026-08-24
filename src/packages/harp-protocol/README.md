# harp-protocol

[![PyPI version](https://badge.fury.io/py/harp-protocol.svg)](https://badge.fury.io/py/harp-protocol)

The Harp Protocol is a binary communication protocol created in order to facilitate and unify the interaction between different devices. It was designed with efficiency and ease of parsing in mind.

For more detail please check the [official Harp Tech documentation](https://harp-tech.org/protocol/BinaryProtocol-8bit.html).

`harp-protocol` provides the building blocks: message framing and the typed register/payload DSL. Each register knows how to build (`format`) and decode (`parse`) its frames.

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

Numpy scalars behave like plain Python numbers in arithmetic, comparison and formatting. Use `int()` or `float()` where a built-in type is required.

It carries no transport or device logic. See [`harp-device`](https://github.com/harp-tech/python/tree/main/src/packages/harp-device) for the device layer.
