"""Serial transport and factory for Harp devices."""

from typing import TypeVar, overload

import serial

from harp.device.client import Device, TransportError
from harp.device.schema import DeviceModuleLike

DEFAULT_BAUDRATE: int = 1_000_000

D = TypeVar("D", bound=Device)
M = TypeVar("M", bound=DeviceModuleLike)


class SerialTransport:
    """A serial-port :class:`~harp.device.client.ITransport` (structural conformance)."""

    def __init__(self, port: str, baudrate: int = DEFAULT_BAUDRATE) -> None:
        self._port = port
        self._baudrate = baudrate
        self._serial: serial.Serial | None = None

    def open(self) -> None:
        try:
            self._serial = serial.Serial(self._port, self._baudrate, timeout=0.1)
            self._serial.dtr = True
        except serial.SerialException as exc:
            raise TransportError(f"Failed to open serial port {self._port!r}: {exc}") from exc

    def write(self, data: bytes) -> None:
        assert self._serial is not None
        try:
            self._serial.write(data)
        except serial.SerialException as exc:
            raise TransportError(str(exc)) from exc

    def read(self) -> bytes:
        assert self._serial is not None
        try:
            waiting = self._serial.in_waiting
            return self._serial.read(waiting if waiting > 0 else 1)
        except serial.SerialException as exc:
            raise TransportError(str(exc)) from exc

    def close(self) -> None:
        if self._serial is None:
            return
        try:
            self._serial.dtr = False
        except Exception:
            pass
        self._serial.close()


@overload
def open_device(
    device_or_module: type[D],
    *,
    port: str,
    baudrate: int = ...,
    raise_on_error: bool = ...,
) -> D: ...


@overload
def open_device(
    device_or_module: M,
    *,
    port: str,
    baudrate: int = ...,
    raise_on_error: bool = ...,
) -> Device[M]: ...


@overload
def open_device(
    device_or_module: None = ...,
    *,
    port: str,
    baudrate: int = ...,
    raise_on_error: bool = ...,
) -> Device[None]: ...


def open_device(
    device_or_module: DeviceModuleLike | type[D] | None = None,
    *,
    port: str,
    baudrate: int = DEFAULT_BAUDRATE,
    raise_on_error: bool = True,
) -> Device:
    """Build a :class:`~harp.device.client.Device` over a serial transport and open it.

    Accepts either a device module or a :class:`~harp.device.client.Device` subclass:

    - **Module** (preferred): validates identity on open::

        from harp.device import behavior, core

        with open_device(behavior, port="COM3") as dev:
            dev.read(core.WhoAmI)           # a core register
            dev.read(behavior.AnalogData)   # declared by the schema

    - **Device subclass**: instantiates the subclass directly, preserving its type::

        with open_device(MyBehavior, port="COM3") as dev:
            dev.arm()   # method defined on MyBehavior

    Omit the first argument for schema-free access, which skips the identity check.

    Like the builtin :func:`open`, the returned device is already connected; use it
    directly or in a ``with`` block for guaranteed close.
    """
    transport = SerialTransport(port, baudrate)
    if isinstance(device_or_module, type):
        return device_or_module(transport, raise_on_error=raise_on_error).open()
    return Device(transport, device_or_module, raise_on_error=raise_on_error).open()
