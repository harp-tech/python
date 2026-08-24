<p align="center">
  <img src="https://raw.githubusercontent.com/harp-tech/python/c0c8c23cc1a965c41834a106fe686be76c2f0520/docs/assets/logo.svg" alt="Harp logo" width="400">
</p>

# harp

This project includes four main packages:

 - **harp-protocol**: Provides the core protocol definitions and utilities for the Harp protocol. See [Protocol API Documentation](https://harp-tech.org/python/api/protocol) for details.

 - **harp-serial**: Implements serial communication functionalities for generic Harp devices. See [Serial API Documentation](https://harp-tech.org/python/api/serial) for more information.

 - **harp-device**: Implements the transport-agnostic `Device` interface, the common register map, and the shared registers and enums. See [Device API Documentation](https://harp-tech.org/python/api/device) for details.

 - **harp-data**: Parses register binary dumps into pandas DataFrames. See [Data API Documentation](https://harp-tech.org/python/api/data) for more information.

## Installation

All packages are published to PyPI. The `harp` package is a metadata package with no code of its own. It depends on the four packages above, so it is the easiest way to get everything:

```sh
pip install harp
```

```sh
uv add harp
```

To install only part of the stack, for example when parsing offline data dumps with no need for serial I/O, install the individual packages. Each one only pulls in what it actually depends on:

| Package | Provides | Depends on |
| --- | --- | --- |
| `harp-protocol` | Core protocol types: registers, messages, payload parsing | none |
| `harp-device` | Transport-agnostic `Device` class, common register map | `harp-protocol` |
| `harp-serial` | Serial COM or tty transport for `Device` | `harp-protocol`, `harp-device` |
| `harp-data` | Parse register binary dumps into pandas DataFrames | `harp-protocol` |

```sh
pip install harp-protocol
pip install harp-device
pip install harp-serial
pip install harp-data
```

`harp-benchmarks`, under `src/packages/`, is internal-only and is never published to PyPI.

## Quickstart

There are two typical ways to use `harp`: talking to a **live device** over a serial connection, or reading **data recorded to disk**.

**Talk to a live device.** Open a connection and read/write registers by class:

```python
from harp import serial
from harp.device import behavior, core

# Use "COMx" on Windows, "/dev/ttyUSBx" on Linux.
with serial.open_device(behavior, port="COM3") as device:
    print(device.read(core.WhoAmI).payload)         # a common register
    print(device.read(behavior.AnalogData).payload) # a device register
    device.write(
        core.OperationControl,
        core.OperationControlPayload(operation_mode=core.OperationMode.ACTIVE),
    )
```

**Read a recorded session.** Point a `DatasetReader` at a dataset folder and read registers into pandas DataFrames, with no hardware required:

```python
from harp import data

# Finds device.yml in the folder, builds the device, returns a ready-to-use reader
reader = data.open_dataset("session.harp")
df = reader.read("AnalogData")  # by name
df = reader.read(44)            # or by address

# `contents` names every register the folder holds
frames = {name: reader.read(name) for name in reader.contents}
```

Given a device package already in hand, pass it as the second argument and read by register class. This is the form that type-checks, and it also checks the device identity against the `device.yml` in the folder:

```python
from harp import data
from harp.device import behavior

reader = data.open_dataset("session.harp", behavior)
df = reader.read(behavior.AnalogData)
```

Both paths are based on a device schema. Given only a `device.yml` and no pre-generated package, `create_device_module` compiles it into a module of register classes at runtime, with no code-generation step. This is exactly what `open_dataset` does internally:

```python
from pathlib import Path

from harp.device import schema

behavior = schema.create_device_module(Path("device.yml").read_bytes())
AnalogData = behavior.AnalogData                 # registers are reached by name
assert behavior.REGISTER_MAP[44] is AnalogData   # or by address
```

See the [Examples](https://harp-tech.org/python/examples/) for the full walkthroughs, including subscribing to device events and working with custom interface-type converters.

## Contributing

harp is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/): every package under `src/packages/` is its own distribution, plus the root `harp` metadata package. Bug reports and contributions are welcome, so please open an issue or pull request.

Clone the repository and install everything with the `dev` dependency group: all workspace packages, editable, plus test and lint tooling.

```sh
uv sync --group dev
```

Before opening a pull request, run the same checks CI runs:

```sh
uv run ruff format --check   # formatting
uv run ruff check            # lint
uv run pyright               # type checking
uv run codespell             # spelling
uv run pytest --cov harp     # tests
```

To add a new package, place it under `src/packages/<name>/` with its own `pyproject.toml` and add it to `[tool.uv.sources]` in the root `pyproject.toml`. If it should ship as part of `harp`, add it to the dependencies of the root package as well.

## Building the documentation

Install the docs dependency group and run mkdocs through uv:

```sh
uv sync --group docs --group dev
uv run mkdocs serve   # live-reloading local preview
uv run mkdocs build   # static site in ./site
```
