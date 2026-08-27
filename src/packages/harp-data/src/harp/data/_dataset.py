import re
from collections.abc import Callable, Mapping
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Generic, TypeVar, overload

import pandas as pd
from harp.device.schema import (
    DEVICE_SCHEMA_FILENAME,
    DeviceModule,
    DeviceModuleLike,
    create_device_module,
    parse_device_schema,
)
from harp.protocol import RegisterBase

from ._reader import parse_to_dataframe

M = TypeVar("M", bound=DeviceModuleLike)

RegisterKey = type[RegisterBase[Any]] | int | str

FileNameResolver = Callable[[Path, str], Mapping[int, list[Path]]]


def default_file_resolver(root: Path, name: str) -> dict[int, list[Path]]:
    """Harp file format resolver: map address -> sorted ``<name>_<address>...`` files."""
    pattern = re.compile(rf"^{re.escape(name)}_(\d+)(?:_.*)?$")
    files: dict[int, list[Path]] = {}
    for path in sorted(root.glob("*.bin")):
        match = pattern.match(path.stem)
        if match is not None:
            files.setdefault(int(match.group(1)), []).append(path)
    return dict(sorted(files.items()))


class DatasetReader(Generic[M]):
    """Reader over a de-multiplexed Harp dataset folder.

    Construct from a device module and a dataset folder, then read the frames of a
    register into a DataFrame by register class, by name, or by address::

        reader = DatasetReader(behavior, "session.harp")
        df = reader.read(behavior.AnalogData)  # by register class
        df = reader.read("AnalogData")         # by name
        df = reader.read(44)                   # by address

    :func:`open_dataset` builds one for a folder that carries its own ``device.yml``.
    :attr:`contents` lists what was recorded, keyed by register name.

    ``device_module`` is a device module -- a generated device package, or one built from a
    schema with :func:`~harp.device.schema.create_device_module`. Its ``REGISTER_MAP`` is
    read at construction.

    The files are matched by a ``<DeviceName>`` prefix, taken from the ``DEVICE_NAME``
    declared by the module. Pass ``name`` to override it, or to supply one when the module
    declares an empty name.

    When the folder carries a ``device.yml`` and the module declares an identity, their
    ``whoAmI`` values are checked against each other. A module paired with the wrong
    folder then fails here rather than decoding the files against the wrong register
    map. ``validate`` turns off every check the reader performs, so a folder whose
    ``device.yml`` is damaged can be read with a module obtained elsewhere.

    The reader is typed on the module it was given, so registers stay accessible through
    :attr:`device_module` at whatever precision that module offers.

    File resolution defaults to the Harp file format: ``<name>_<address>.bin`` and,
    when a register was logged as several ``<name>_<address>_<suffix>.bin`` chunks,
    they are concatenated in filename order. Pass ``resolver`` (a :data:`FileNameResolver`)
    to support an alternative on-disk layout.

    ``epoch`` anchors the time index of every read to absolute time, so one dataset is
    read on one clock rather than the choice being made per register. It describes how
    the recording was anchored. The reference Harp clock starts at :data:`REFERENCE_EPOCH`.
    """

    def __init__(
        self,
        device_module: M,
        root: str | PathLike[str],
        *,
        name: str | None = None,
        resolver: FileNameResolver = default_file_resolver,
        epoch: datetime | None = None,
        validate: bool = True,
    ) -> None:
        self._device_module = device_module
        self._root = Path(root)
        self._name_override = name
        self._resolver = resolver
        self._epoch = epoch
        self._name = self._resolve_name()
        self._paths = dict(self._resolver(self._root, self._name))
        registers = device_module.REGISTER_MAP
        self._name_map = {registers[address].__name__: address for address in sorted(registers)}
        if validate:
            self._validate_whoami()

    @property
    def root(self) -> Path:
        """The dataset folder being read."""
        return self._root

    @property
    def device_module(self) -> M:
        """The device module this reader parses against, as the type it was given.

        A generated package resolves each register to its own class; one built by
        :func:`~harp.device.schema.create_device_module` resolves them collectively,
        the same ceiling as accessing it directly.
        """
        return self._device_module

    @property
    def name(self) -> str:
        """The ``<DeviceName>`` prefix used to match binary files."""
        return self._name

    def _validate_whoami(self) -> None:
        expected = self._device_module.WHO_AM_I
        if expected == 0:
            return
        path = self._root / DEVICE_SCHEMA_FILENAME
        if not path.is_file():
            return
        try:
            actual = parse_device_schema(path.read_bytes()).whoAmI
        except ValueError:
            return
        if actual is None or int(actual) == expected:
            return
        raise ValueError(
            f"WhoAmI mismatch: {self._name} expects 0x{expected:04x} but the schema in "
            f"{self._root} declares 0x{int(actual):04x}."
        )

    def _resolve_name(self) -> str:
        if self._name_override is not None:
            return self._name_override
        declared = self._device_module.DEVICE_NAME
        if declared:
            return declared
        raise ValueError(
            f"The device module declares an empty DEVICE_NAME, so it cannot name the "
            f"files under {self._root}. Pass name= with the file prefix to read."
        )

    @property
    def contents(self) -> Mapping[str, int]:
        """The mapping from register name to address for registers with data under :attr:`root`."""
        return {name: address for name, address in self._name_map.items() if address in self._paths}

    @property
    def paths(self) -> Mapping[int, list[Path]]:
        """The mapping from address to binary files discovered under :attr:`root`."""
        return self._paths

    def read(
        self,
        register: RegisterKey,
        *,
        suffix: str | None = None,
        time_index: bool = True,
        keep_type: bool = False,
        decode_enums: bool = True,
        demux_bit_masks: bool = False,
    ) -> pd.DataFrame:
        """Read the data of one register into a DataFrame.

        ``register`` is a register class, a register name, or an address. Names are
        resolved through the device register map rather than the module namespace, which
        declares no core registers. Prefer the class where a generated package supplies
        one, since it is the only form a type checker can verify. A module built by
        :func:`~harp.device.schema.create_device_module` resolves its registers as
        ``Any``, so there the generated module verifies no more than the name does.

        A register declared in the device register map with no data present in the folder
        reads as an empty DataFrame carrying the same columns, since the schema describes
        the structure of the data regardless of whether anything was recorded.
        :attr:`contents` is what tells the two cases apart. A register the device does not
        declare at all raises ``KeyError``.

        ``suffix`` selects a single ``<name>_<address>_<suffix>.bin`` chunk, and naming
        one that is absent raises ``FileNotFoundError`` (default: concatenate every
        chunk for the address). The remaining options match
        :func:`~harp.data.parse_to_dataframe`, except that the epoch is the one the
        reader was opened with.
        """
        cls, address = self._resolve(register)
        paths = self._resolve_paths(address, suffix)
        raw = b"".join(p.read_bytes() for p in paths)
        return parse_to_dataframe(
            cls,
            raw,
            time_index=time_index,
            epoch=self._epoch,
            keep_type=keep_type,
            decode_enums=decode_enums,
            demux_bit_masks=demux_bit_masks,
        )

    def _resolve(self, register: RegisterKey) -> tuple[type[RegisterBase[Any]], int]:
        if isinstance(register, type):
            return register, register.address
        registers = self._device_module.REGISTER_MAP
        if isinstance(register, str):
            address = self._name_map.get(register)
            if address is None:
                raise KeyError(f"No register named {register!r} in the map of this device.")
            return registers[address], address
        cls = registers.get(register)
        if cls is None:
            raise KeyError(f"No register at address {register} in the map of this device.")
        return cls, register

    def _resolve_paths(self, address: int, suffix: str | None) -> list[Path]:
        paths = self._paths.get(address) or []
        if suffix is not None:
            paths = [p for p in paths if p.stem.endswith(f"_{suffix}")]
            if not paths:
                raise FileNotFoundError(
                    f"No '_{suffix}' chunk for register address {address} under {self._root}."
                )
        return paths


@overload
def open_dataset(
    root: str | PathLike[str],
    device_module: M,
    *,
    name: str | None = ...,
    resolver: FileNameResolver = ...,
    epoch: datetime | None = ...,
    validate: bool = ...,
) -> DatasetReader[M]: ...


@overload
def open_dataset(
    root: str | PathLike[str],
    device_module: None = ...,
    *,
    schema: str | PathLike[str] | None = ...,
    name: str | None = ...,
    resolver: FileNameResolver = ...,
    converters: Mapping[str, Any] | None = ...,
    require_converters: bool = ...,
    epoch: datetime | None = ...,
    validate: bool = ...,
) -> DatasetReader[DeviceModule]: ...


def open_dataset(
    root: str | PathLike[str],
    device_module: DeviceModuleLike | None = None,
    *,
    schema: str | PathLike[str] | None = None,
    name: str | None = None,
    resolver: FileNameResolver = default_file_resolver,
    converters: Mapping[str, Any] | None = None,
    require_converters: bool = True,
    epoch: datetime | None = None,
    validate: bool = True,
) -> DatasetReader:
    """Open a de-multiplexed Harp dataset folder and return a :class:`DatasetReader`.

    If the device module is omitted, the schema file inside the folder will be used.
    The ``device.yml`` inside ``root`` is first built into a module using
    :func:`~harp.device.schema.create_device_module`.

    If a device module is provided, its identity class will be used to validate the
    dataaset, and a generated package additionally carries register classes a
    type checker can verify.

    ``schema`` points at the schema file when it isn't ``root/device.yml``, and
    ``converters`` and ``require_converters`` are forwarded to
    :func:`~harp.device.schema.create_device_module` for custom ``interfaceType``
    decoding. These three parameters describe alternative ways to supply a module, so
    they are mutually exclusive, and will raise when more than one is specified.

    ``epoch`` anchors the time index of every read to absolute time, since the anchor
    describes the recording rather than one register. The reference Harp clock starts
    at :data:`REFERENCE_EPOCH`, and the default of ``None`` gives float seconds.

    ``validate`` cannot rescue a corrupt ``device.yml`` if that schema file is also
    used to build the module. Reading such a folder always requires supplying a module
    obtained elsewhere.
    """
    root_path = Path(root)
    if device_module is not None:
        if schema is not None or converters is not None:
            raise TypeError(
                "schema= and converters= describe how to build a device module, so they "
                "do not apply when one is given. Drop them, or drop the device module."
            )
        return DatasetReader(
            device_module,
            root_path,
            name=name,
            resolver=resolver,
            epoch=epoch,
            validate=validate,
        )
    schema_path = Path(schema) if schema is not None else root_path / DEVICE_SCHEMA_FILENAME
    if not schema_path.is_file():
        raise FileNotFoundError(
            f"No device schema at '{schema_path}'. Pass schema= to point at a device.yml, "
            f"or pass the device module itself as open_dataset(root, device_module)."
        )
    built = create_device_module(
        schema_path.read_text(), converters=converters, require_converters=require_converters
    )
    return DatasetReader(
        built,
        root_path,
        name=name,
        resolver=resolver,
        epoch=epoch,
        validate=validate and schema is not None,
    )
