"""Record what a device emits as a de-multiplexed dataset folder.

The layout the Harp file format standard defines, one binary file per register beside
the schema of the device::

    session.harp/
      Behavior_0.bin
      Behavior_44.bin
      ...
      device.yml

Each message is appended to the file of its own register as the complete Harp frame it
arrived as, header and checksum included, so a file is a run of frames that needs nothing
else to be read. This is what the reference C# writer records.

Writing is a device concern rather than an analysis one, so it lives here and costs no
pandas. :class:`~harp.data.DatasetReader` is the other end, reading such a folder into
DataFrames.
"""

import threading
from collections.abc import Callable, Iterable
from functools import partial
from os import PathLike
from pathlib import Path
from typing import IO, Any, Self

from harp.device.schema import DEVICE_SCHEMA_FILENAME, DeviceModuleLike
from harp.protocol import HarpMessage, MessageType

from ._device import Device, Subscription

_ALL_MESSAGE_TYPES: frozenset[MessageType] = frozenset(MessageType)
"""Every message type, which is what a recording captures by default.

A Harp device answers a read of one register with a ``Read`` message and a write with a
``Write`` message, and the register dump a device performs on request arrives as a burst
of ``Read`` messages. Capturing only ``Event`` would drop the configuration the device
was running under, so a recording captures all three.
"""


_FileNameFormatter = Callable[[int], str]
"""The file one register is written to, named relative to the dataset folder.
"""


def _default_file_formatter(device_name: str, address: int) -> str:
    """Harp file format name for one register: ``<device_name>_<address>.bin``.

    The standard also allows a trailing ``_<suffix>`` field, which a custom formatter
    supplies; this names the plain form. :func:`~harp.data.default_file_resolver` reads
    either back. :class:`DeviceWriter` binds ``device_name`` to reach the
    shape a formatter has, so what it calls per register takes the address alone.
    """
    return f"{device_name}_{address}.bin"


class DeviceWriter:
    """De-multiplexing sink writing Harp messages into a dataset folder.

    Hand it messages with :meth:`write`, or let :func:`attach_writer` feed it from a live
    device. The file of a register is created the first time a message for that register
    arrives, so the folder holds exactly the registers that were seen::

        with DeviceWriter(behavior, "session.harp") as writer:
            writer.write(message)

    ``device_module`` supplies the ``<DeviceName>`` prefix of every file and the
    ``device.yml`` written into the folder. That copy is not optional: a recording
    describes itself or it is not written at all, so a module carrying no
    ``DEVICE_METADATA`` raises here rather than leaving a folder nothing can be decoded
    against later.

    The ``<DeviceName>`` prefix is the ``DEVICE_NAME`` of that module and nothing else,
    so a recording is named after the device that produced it and a reader matching files
    by that name cannot be pointed at the wrong prefix.

    ``overwrite`` decides what happens when a file is already there. By default the first
    write to an existing path raises :class:`FileExistsError`, so a folder is never
    silently half-overwritten by a second recording. A register whose file is created
    late in the run therefore fails late in the run, rather than at construction.

    File naming defaults to the Harp file format, ``<DeviceName>_<address>.bin``. Pass
    ``formatter`` -- any callable taking an address and returning a name relative to the
    folder -- for any other layout, paired with the
    :data:`~harp.data.FileNameResolver` that reads it back. It names one file from an
    address alone, so the device name is what the default is bound with rather than
    something a custom formatter is handed. A custom one holds whatever else it needs --
    the standard trailing ``_<suffix>`` field, a timestamp, a subfolder -- and a module
    that names no device is then enough to record with::

        DeviceWriter(behavior, root, formatter=lambda address: f"Behavior_{address}_0.bin")

    The folder itself is created eagerly, along with its ``device.yml``, so a path that
    cannot be written fails before any message is taken. Writes are serialized and may
    come from any thread, and the frames of one register keep the order they were written
    in. Files stay open and buffered until :meth:`flush` or :meth:`close`, so use the
    writer as a context manager or close it yourself, or a recording ends short of its
    last frames.
    """

    def __init__(
        self,
        device_module: DeviceModuleLike,
        root: str | PathLike[str],
        *,
        formatter: _FileNameFormatter | None = None,
        overwrite: bool = False,
    ) -> None:
        self._root = Path(root)
        self._name = self._resolve_name(device_module, required=formatter is None)
        self._formatter = formatter or partial(_default_file_formatter, self._name)
        self._overwrite = overwrite
        self._lock = threading.Lock()
        self._streams: dict[int, IO[bytes]] = {}
        self._paths: dict[int, Path] = {}
        self._subscription: Subscription | None = None
        self._closed = False
        self._root.mkdir(parents=True, exist_ok=True)
        self._schema_path = self._write_schema(device_module)

    def _resolve_name(self, module: DeviceModuleLike, *, required: bool) -> str:
        """The device name, demanded only when the default formatter has to build on it."""
        declared = module.DEVICE_NAME
        if declared or not required:
            return declared
        raise ValueError(
            f"No name for the files under {self._root}: this module declares an empty "
            f"DEVICE_NAME, so only a formatter of your own can name them."
        )

    def _write_schema(self, module: DeviceModuleLike) -> Path:
        """Copy the schema of ``module`` into the folder, so the folder describes itself."""
        schema = getattr(module, "DEVICE_METADATA", b"")
        if not schema:
            raise ValueError(
                f"No DEVICE_METADATA to describe {self._root} with. Record with a module "
                f"declaring one."
            )
        path = self._root / DEVICE_SCHEMA_FILENAME
        if path.exists() and not self._overwrite:
            raise FileExistsError(f"'{path}' already exists. Pass overwrite=True to replace it.")
        path.write_bytes(schema)
        return path

    @property
    def root(self) -> Path:
        """The dataset folder being written."""
        return self._root

    @property
    def name(self) -> str:
        """The ``<DeviceName>`` prefix every file is written under."""
        return self._name

    @property
    def paths(self) -> dict[int, Path]:
        """The file written for each register address seen so far."""
        with self._lock:
            return dict(sorted(self._paths.items()))

    @property
    def schema_path(self) -> Path:
        """The ``device.yml`` written into the folder."""
        return self._schema_path

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run."""
        return self._closed

    def write(self, message: HarpMessage[Any]) -> None:
        """Append the complete frame of ``message`` to the file of its register.

        Raises :class:`ValueError` once the writer is closed, and
        :class:`FileExistsError` when this is the first message for a register whose file
        is already on disk and ``overwrite`` is off.
        """
        frame = message.bytes
        address = message.address
        with self._lock:
            if self._closed:
                raise ValueError(f"This writer for {self._root} is closed.")
            stream = self._streams.get(address)
            if stream is None:
                stream = self._open(address)
            stream.write(frame)

    def _open(self, address: int) -> IO[bytes]:
        path = self._root / self._formatter(address)
        path.parent.mkdir(parents=True, exist_ok=True)  # a formatter may name a subfolder
        stream = open(path, "wb" if self._overwrite else "xb")
        self._streams[address] = stream
        self._paths[address] = path
        return stream

    def _deliver(self, message: HarpMessage[Any]) -> None:
        """Take a message off a subscription, dropping one still in flight at close.

        Unlike :meth:`write` this cannot raise on a closed writer, since a message may
        already be on its way to a handler when the subscription is cancelled, and there
        is nothing wrong with the recording having ended first.
        """
        if self._closed:
            return
        self.write(message)

    def flush(self) -> None:
        """Push every buffered frame to the operating system."""
        with self._lock:
            streams = list(self._streams.values())
        for stream in streams:
            stream.flush()

    def close(self) -> None:
        """Detach from the device, if attached, and close every open file. Idempotent."""
        subscription = self._subscription
        if subscription is not None:
            subscription.unsubscribe()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._subscription = None
            streams = list(self._streams.values())
            self._streams.clear()
        for stream in streams:
            stream.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def attach_writer(
    device: Device[Any],
    root: str | PathLike[str],
    *,
    formatter: _FileNameFormatter | None = None,
    message_types: MessageType | Iterable[MessageType] = _ALL_MESSAGE_TYPES,
    overwrite: bool = False,
) -> DeviceWriter:
    """Record everything ``device`` emits into a dataset folder at ``root``.

    Builds a :class:`DeviceWriter` from the module the device was constructed with and
    subscribes it to the whole message stream, returning the writer so the recording can
    be flushed, inspected and ended::

        with Device(transport, behavior) as dev, attach_writer(dev, "session.harp"):
            ...  # every message the device emits is recorded

    The writer owns the subscription, so closing it, by leaving the ``with`` block or by
    calling :meth:`DeviceWriter.close`, both detaches from the device and closes the
    files. Nothing else has to be unsubscribed, and the device outlives the recording, so
    several recordings may be made over one open device.

    ``message_types`` narrows what is recorded, and records all of them by default: a
    Harp device answers reads and writes with ``Read`` and ``Write`` messages, and the
    register dump it performs on request arrives as ``Read``, so recording only ``Event``
    would drop the configuration the device was running under. Requesting that dump is
    the caller's to do, by writing to :class:`~harp.device.core.OperationControl` with
    ``dump_registers`` set, once the writer is attached.

    The remaining arguments are those of :class:`DeviceWriter`. The device has to have
    been constructed with a module, since that module is what names the files and
    describes the recording; one opened without it cannot be recorded from.
    """
    if device.module is None:
        raise ValueError(
            "This device was opened without a module, so nothing names or describes a "
            "recording of it. Construct it with the device module to record."
        )
    writer = DeviceWriter(
        device.module,
        root,
        formatter=formatter,
        overwrite=overwrite,
    )
    try:
        writer._subscription = device.subscribe_all(writer._deliver, message_types=message_types)
    except Exception:
        writer.close()
        raise
    return writer
