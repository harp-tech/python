"""Emit a Python module of register classes from a device schema.

A generated device package is already a module: register classes at module level
and a ``REGISTER_MAP`` beside them (see the ``harp-device`` README).
:func:`create_device_module` builds that same shape at runtime from a ``device.yml``, so a
schema-driven device and a generated one are reached the same way, by name from the
module or by address through ``REGISTER_MAP``.
"""

import types
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from harp.protocol import RegisterBase

from harp.device.core import REGISTER_MAP as CORE_REGISTER_MAP
from ._emit import ConverterValue, _Emitter, parse_device_schema

_DEFAULT_NAME = "Device"
"""Module name used when the schema carries no ``device`` header."""


@runtime_checkable
class DeviceModuleLike(Protocol):
    """Any module describing a device, however it was produced.

    A generated device package is a plain module, so it cannot be named by a class;
    what identifies it is describing a device. Matching structurally accepts both it
    and :class:`DeviceModule`, and rejects the core register set, which carries
    registers but is not a device.

    ``DEVICE_NAME`` is required rather than optional, so a generated package always
    states the name used for its recordings. A schema declaring none still
    builds, since :class:`DeviceModule` declares the member and leaves it empty.
    """

    DEVICE_NAME: str
    WHO_AM_I: int
    REGISTER_MAP: dict[int, type[RegisterBase[Any]]]


class DeviceModule(types.ModuleType):
    """The type of the module returned by :func:`create_device_module`.

    The declarations of the schema are reached by name and typed ``Any``, since they
    exist only at runtime. ``DEVICE_NAME``, ``REGISTER_MAP``, ``WHO_AM_I`` and
    ``__all__`` are declared here and carry their own types.
    """

    DEVICE_NAME: str
    """The device name declared by the schema. Empty when absent."""

    WHO_AM_I: int
    """The device identity declared by the schema. ``0`` when absent."""

    REGISTER_MAP: dict[int, type[RegisterBase[Any]]]
    """Address -> register class, the core Harp registers merged with those of the schema."""

    __all__: list[str]
    """The declarations of the schema, beside ``REGISTER_MAP`` and ``WHO_AM_I``."""


def create_device_module(
    text: str | bytes,
    *,
    name: Optional[str] = None,
    converters: Optional[Mapping[str, ConverterValue]] = None,
    require_converters: bool = True,
) -> DeviceModule:
    """Emit a module of register classes from ``device.yml`` text.

    The module names what the schema declares, its registers beside the enums and
    payload classes they are built from, so ``behavior.AnalogData``,
    ``behavior.AnalogDataPayload`` and ``behavior.EncoderModeMask`` all resolve while a
    core register such as ``WhoAmI`` is imported from :mod:`harp.device.core`, keeping
    one definition of each. This is the same set a generated device package holds. A
    name describing two declarations is rejected rather than shadowed. Beside them
    it holds:

    * ``REGISTER_MAP``, the device address space, so the core registers are
      present here even though the module does not name them;
    * ``WHO_AM_I``, the identity declared by the schema (``0`` for an unregistered device);
    * ``DEVICE_NAME``, the ``device`` name of the schema, or ``name`` when given, and
      empty for a header-less register fragment. Recordings are written under this
      name, so :class:`~harp.data.DatasetReader` matches files by it;
    * ``__name__``, the same name, falling back to ``"Device"`` so the module is never
      anonymous. This names the module rather than the device, and is not part of what
      a device module promises;
    * ``__doc__``, the optional ``description`` of the schema.

    Because the names come from the schema at runtime they don't autocomplete, and
    each resolves as ``Any`` rather than its own type. A generated device package is
    a real module on disk and gives both. On an address clash the device register
    replaces the core one in ``REGISTER_MAP``.

    ``text`` is the schema itself rather than a path to it, matching
    :func:`parse_device_schema`, so read the file first. The module is **not**
    registered in :data:`sys.modules`, so it cannot be reached by ``import`` and two
    schemas may share a name without clashing. Bind it yourself::

        behavior = create_device_module(Path("device.yml").read_bytes())
        behavior.AnalogData
    """
    device = parse_device_schema(text)
    emitter = _Emitter(device, converters, require_converters)
    registers = emitter.emit()
    device_name = name or device.device or ""
    module_name = device_name or _DEFAULT_NAME

    contents: dict[str, Any] = {**emitter.enums, **emitter.payloads, **registers}
    register_map = {cls.address: cls for cls in CORE_REGISTER_MAP.values()}
    register_map.update({cls.address: cls for cls in registers.values()})

    for declaration in contents.values():
        declaration.__module__ = module_name

    module = DeviceModule(module_name, device.description)
    vars(module).update(
        contents,
        DEVICE_NAME=device_name,
        REGISTER_MAP=register_map,
        WHO_AM_I=int(device.whoAmI or 0),
        __all__=[*sorted(contents), "DEVICE_NAME", "REGISTER_MAP", "WHO_AM_I"],
    )
    return module
