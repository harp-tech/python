import queue
import threading
import types
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from harp.device import core
from harp.device.client import Device, DeviceError, TransportError
from harp.protocol import MessageType
from tests.fixtures import make_frame_from_raw

_U16 = 0x02
"""Payload-type byte of a U16 payload, as the size nibble alone."""


class _NullTransport:
    def open(self) -> None: ...

    def close(self) -> None: ...

    def write(self, data: bytes) -> None: ...

    def read(self) -> bytes:
        return b""


class _ScriptedTransport:
    """A transport replying with whatever ``on_write`` returns for each request.

    Frames are queued from inside ``write``, which is reached only once the request is
    registered, so a reply cannot be dispatched before there is a waiter to receive it.
    Setting ``failing`` makes the next read fail, as a removed port would.
    """

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.failing = False
        self.on_write: Callable[[bytes], Iterable[bytes]] | None = None
        self._inbox: queue.SimpleQueue[bytes] = queue.SimpleQueue()

    def open(self) -> None: ...

    def close(self) -> None: ...

    def write(self, data: bytes) -> None:
        self.writes.append(data)
        if self.on_write is not None:
            for frame in self.on_write(data):
                self._inbox.put(frame)

    def read(self) -> bytes:
        if self.failing:
            raise TransportError("simulated transport failure")
        try:
            return self._inbox.get(timeout=0.01)
        except queue.Empty:
            return b""


class _ShortTimeoutDevice(Device[None]):
    """A device that gives up on a reply quickly, so a test never waits five seconds."""

    REPLY_TIMEOUT = 0.5


def _module(name: str, **attrs: object) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.DEVICE_NAME = name
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def test_zero_whoami_skips_validation():
    # 0 marks an unregistered device, so opening must not read WhoAmI at all.
    device = Device(_NullTransport(), _module("Unregistered", WHO_AM_I=0, REGISTER_MAP={}))
    with device:
        assert device.module.WHO_AM_I == 0


def test_module_property_returns_given_module():
    module = _module("Behavior", WHO_AM_I=0, REGISTER_MAP={})
    assert Device(_NullTransport(), module).module is module
    assert Device(_NullTransport()).module is None


def test_write_reply_does_not_satisfy_read():
    # A reply carries the message type of its request, so a write reply on the same
    # register must not answer a pending read.
    transport = _ScriptedTransport()
    transport.on_write = lambda _: (
        core.WhoAmI.format(np.uint16(7), message_type=MessageType.Write),
        core.WhoAmI.format(np.uint16(9), message_type=MessageType.Read),
    )
    with _ShortTimeoutDevice(transport) as device:
        assert int(device.read(core.WhoAmI).payload) == 9


def test_event_does_not_satisfy_read():
    # Events are unsolicited, so one for the same register must not answer a read.
    transport = _ScriptedTransport()
    transport.on_write = lambda _: (
        core.WhoAmI.format(np.uint16(7), message_type=MessageType.Event),
        core.WhoAmI.format(np.uint16(9), message_type=MessageType.Read),
    )
    with _ShortTimeoutDevice(transport) as device:
        assert int(device.read(core.WhoAmI).payload) == 9


def test_concurrent_reads_share_one_reply():
    # The wire carries no request identifier, so two reads in flight on one register
    # cannot be told apart in the reply. Both are answered by it rather than one
    # replacing the waiter of the other.
    transport = _ScriptedTransport()
    both_in_flight = threading.Barrier(2)

    def reply_once(_: bytes) -> Iterable[bytes]:
        first = both_in_flight.wait(timeout=2) == 0
        return () if first else (core.WhoAmI.format(np.uint16(9), message_type=MessageType.Read),)

    transport.on_write = reply_once
    with _ShortTimeoutDevice(transport) as device, ThreadPoolExecutor(2) as pool:
        replies = [pool.submit(device.read, core.WhoAmI) for _ in range(2)]
        assert [int(reply.result(timeout=2).payload) for reply in replies] == [9, 9]


def test_transport_failure_faults_pending_read():
    transport = _ScriptedTransport()

    def fail(_: bytes) -> Iterable[bytes]:
        transport.failing = True
        return ()

    transport.on_write = fail
    with _ShortTimeoutDevice(transport) as device:
        with pytest.raises(TransportError):
            device.read(core.WhoAmI)


def test_read_after_transport_failure_raises_transport_error():
    # A failure that has already stopped the reader is reported to the next request
    # rather than leaving it to time out on a reply that cannot arrive.
    transport = _ScriptedTransport()

    def fail(_: bytes) -> Iterable[bytes]:
        transport.failing = True
        return ()

    transport.on_write = fail
    with _ShortTimeoutDevice(transport) as device:
        with pytest.raises(TransportError):
            device.read(core.WhoAmI)
        with pytest.raises(TransportError):
            device.read(core.WhoAmI)
    assert len(transport.writes) == 1  # the second request never reached the transport


def test_error_reply_raises_device_error():
    frame = make_frame_from_raw(MessageType.Read | 0x08, 0, 255, _U16, b"\x07\x00")
    transport = _ScriptedTransport()
    transport.on_write = lambda _: (frame,)
    with _ShortTimeoutDevice(transport) as device:
        with pytest.raises(DeviceError) as error:
            device.read(core.WhoAmI)
    assert error.value.reply.bytes == frame  # the frame is kept, not formatted away


def test_error_reply_returned_when_not_raising():
    frame = make_frame_from_raw(MessageType.Read | 0x08, 0, 255, _U16, b"\x07\x00")
    transport = _ScriptedTransport()
    transport.on_write = lambda _: (frame,)
    with _ShortTimeoutDevice(transport, raise_on_error=False) as device:
        reply = device.read(core.WhoAmI)
    assert reply.has_error
    assert int(reply.payload) == 7
