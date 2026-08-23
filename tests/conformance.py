"""Static conformance checks for the documented public API.

Nothing here runs. Every function is a type-checker fixture, asserting the type a
documented expression resolves to, so a change that silently degrades an inferred
type fails the build rather than being noticed downstream.
"""

from typing import Any, ClassVar, assert_type

import numpy as np
from harp.data import DatasetReader, open_dataset
from harp.device.client import Device, ITransport
from harp.device.core import OperationControl, OperationControlPayload, WhoAmI
from harp.device.schema import DeviceModule, DeviceModuleLike, create_device_module
from harp.protocol import (
    ArrayConverter,
    Field,
    HarpMessage,
    IdentityConverter,
    RegisterBase,
    StructPayload,
)
from harp.serial import open_device
from numpy.typing import NDArray


def schema_built_registers(yml: str) -> None:
    """A module built from a schema types its registers collectively."""
    behavior = create_device_module(yml)
    assert_type(behavior, DeviceModule)
    assert_type(behavior.AnalogData, Any)
    assert_type(behavior.REGISTER_MAP, dict[int, type[RegisterBase[Any]]])
    assert_type(behavior.WHO_AM_I, int)


def statically_declared_registers(device: Device) -> None:
    """A register written out in a module carries its payload type through read."""
    assert_type(device.read(WhoAmI), HarpMessage[np.uint16])
    assert_type(device.read(WhoAmI).payload, np.uint16)
    assert_type(device.read(OperationControl).payload, OperationControlPayload)


def register_writes(device: Device, payload: OperationControlPayload) -> None:
    """Write accepts the payload type its register parses to."""
    assert_type(device.write(OperationControl, payload).payload, OperationControlPayload)


def device_with_module(transport: ITransport, module: DeviceModule) -> None:
    """Device constructed with a module is typed on that module."""
    device = Device(transport, module)
    assert_type(device, Device[DeviceModule])
    assert_type(device.module, DeviceModule)


def device_without_module(transport: ITransport) -> None:
    """Device constructed without a module is Device[None]."""
    device = Device(transport)
    assert_type(device, Device[None])
    assert_type(device.module, None)


def open_device_with_module(module: DeviceModule) -> None:
    """open_device with a module returns Device[M]."""
    device = open_device(module, port="COM3")
    assert_type(device, Device[DeviceModule])
    assert_type(device.module, DeviceModule)


def open_device_without_module() -> None:
    """open_device without a module returns Device[None]."""
    device = open_device(port="COM3")
    assert_type(device, Device[None])
    assert_type(device.module, None)


def open_device_with_subclass() -> None:
    """open_device with a Device subclass preserves its type."""

    class MyDevice(Device[DeviceModule]):
        def arm(self) -> None: ...

    device = open_device(MyDevice, port="COM3")
    assert_type(device, MyDevice)


def dataset_reader_accepts_either_module(
    schema_built: DeviceModule, generated: DeviceModuleLike
) -> None:
    """The reader takes a schema-built module and a generated package alike."""
    DatasetReader(schema_built, "session.harp")
    reader = DatasetReader(generated, "session.harp")
    reader.read(WhoAmI)
    reader.read(44)


def dataset_reader_keeps_module_type(
    schema_built: DeviceModule, generated: DeviceModuleLike
) -> None:
    """The reader is typed on the module it was given, not on the contract.

    Reading it back as ``DeviceModuleLike`` would leave only the three declarations of
    the contract, so every register accessed through the reader would fail to resolve.
    """
    assert_type(DatasetReader(schema_built, "session.harp").device_module, DeviceModule)
    assert_type(DatasetReader(generated, "session.harp").device_module, DeviceModuleLike)


def dataset_reader_registers_resolve_through_the_module() -> None:
    """A register stays accessible through the reader, at the precision of its module.

    A schema-built module resolves collectively, as it does when accessed directly, so
    the ceiling here is the one :func:`create_device_module` documents. A generated
    package carries its own declarations and resolves each to its own class.
    """
    reader = open_dataset("session.harp")
    assert_type(reader, DatasetReader[DeviceModule])
    assert_type(reader.device_module.AnalogData, Any)
    reader.read(reader.device_module.AnalogData)


def open_dataset_keeps_supplied_module_type(generated: DeviceModuleLike) -> None:
    """A module passed through the entry point types the reader on itself."""
    assert_type(open_dataset("session.harp", generated), DatasetReader[DeviceModuleLike])


def open_device_prefers_the_subclass_overload() -> None:
    """A Device subclass is matched as a subclass even when it looks like a module.

    type[D] is narrower than the structural module overload, so it has to come first:
    a class carrying the members of DeviceModuleLike satisfies it too, and the module
    overload would otherwise win and return Device[type[Hybrid]]. The class has to
    carry every member for this to test the ordering rather than the match.
    """

    class Hybrid(Device[None]):
        DEVICE_NAME: ClassVar[str] = "Hybrid"
        WHO_AM_I: ClassVar[int] = 1216
        REGISTER_MAP: ClassVar[dict[int, type[RegisterBase[Any]]]] = {}

    device = open_device(Hybrid, port="COM3")
    assert_type(device, Hybrid)


def array_payload_members() -> None:
    """A member spanning several elements resolves as an array of the element type.

    The equivalent sub-array dtype passed to IdentityConverter produces the same bytes
    but resolves as a scalar, since a sub-array dtype carries np.void as its scalar
    type, so ArrayConverter is what carries the element type through the descriptor.
    """

    class Payload(StructPayload[np.float32], length=6):
        analog0: np.float32 = Field(IdentityConverter(np.float32), offset=0)
        accelerometer: NDArray[np.float32] = Field(ArrayConverter(np.float32, 3), offset=3)

    payload = Payload(analog0=np.float32(0), accelerometer=np.zeros(3, dtype=np.float32))
    assert_type(payload.analog0, np.float32)
    assert_type(payload.accelerometer, NDArray[np.float32])
