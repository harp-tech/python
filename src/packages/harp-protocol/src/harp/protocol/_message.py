"""Harp message container."""

import struct
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast

from typing_extensions import Sentinel

from ._builder import build_message_frame
from ._checksum import validate as _validate_checksum
from ._constants import (
    _DEFAULT_PORT,
    _HEADER_LEN,
    _MIN_FRAME_LEN,
    _TICK_PERIOD_S,
    _TIMESTAMP_FLAG,
    _TIMESTAMP_LEN,
    _TIMESTAMPED_PAYLOAD_OFFSET,
)
from ._message_type import MessageType, _message_type_from_byte_safe
from ._payload import PayloadBase
from ._payload_type import PayloadType, decode_payload_type

P = TypeVar("P")
_P = TypeVar("_P")
_P_co = TypeVar("_P_co", covariant=True)

_UNDECODED = Sentinel("_UNDECODED")
"""Marks a message whose payload no register has decoded yet."""


class HarpParseError(Exception):
    """An exception raised for errors encountered during message parsing"""

    pass


class PayloadDecoder(Protocol[_P_co]):
    """Reads a payload of type ``_P_co`` out of a message.

    Structural rather than nominal, so a message never has to know about registers, and
    anything declaring a payload type, a payload class and a ``parse`` satisfies it. Every
    ``RegisterBase`` does. ``payload_class.payload_dtype`` is what fixes how many payload
    bytes the decoder consumes, and it is the same quantity ``parse`` reads the frame
    with, so the two cannot disagree about the extent of a payload.
    """

    payload_type: ClassVar["PayloadType"]
    payload_class: ClassVar[type[PayloadBase[Any]]]

    @classmethod
    def parse(cls, value: Any) -> _P_co: ...


class HarpMessage(Generic[P]):
    """A Harp message backed by its raw frame bytes, parameterized by its payload type.

    Build with the constructor or parse from wire bytes with ``HarpMessage.parse()``.
    A message off the wire is a ``HarpMessage[Any]``, since a frame declares only how
    its payload is encoded and not which register contract it satisfies. Decoding it
    with a register yields a ``HarpMessage[P]``, whose ``payload`` is that contract.
    """

    __slots__ = ("_bytes", "_payload")

    def __init__(
        self,
        message_type: MessageType,
        address: int,
        payload_type: PayloadType,
        payload_bytes: bytes = b"",
        *,
        port: int = _DEFAULT_PORT,
        timestamp: float | None = None,
    ) -> None:
        self._bytes: bytes = build_message_frame(
            message_type, address, payload_type, payload_bytes, port=port, timestamp=timestamp
        )
        self._payload: P | _UNDECODED = _UNDECODED

    @classmethod
    def parse(cls, data: bytes | bytearray | memoryview) -> "HarpMessage[Any]":
        """Parse and validate a complete Harp Message from a byte sequence. Raises ``HarpParseError`` on failure."""
        raw = data if isinstance(data, bytes) else bytes(data)

        if len(raw) < _MIN_FRAME_LEN:
            raise HarpParseError(f"Frame too short: {len(raw)} bytes (minimum {_MIN_FRAME_LEN})")

        if not _validate_checksum(raw):
            raise HarpParseError("Checksum mismatch")

        # Validate MessageType byte (bits 7,6,5,4,2 must be 0; bits 1:0 are type)
        b0 = raw[0]
        if _message_type_from_byte_safe(b0) is None:
            raise HarpParseError(f"Invalid MessageType byte: 0x{b0:02x}")

        length = raw[1]
        if len(raw) != length + 2:
            raise HarpParseError(f"Length field {length} inconsistent with buffer size {len(raw)}")

        try:
            decode_payload_type(raw[4])
        except ValueError as exc:
            raise HarpParseError(str(exc)) from exc

        if bool(raw[4] & _TIMESTAMP_FLAG) and len(raw) < _HEADER_LEN + _TIMESTAMP_LEN + 1:
            raise HarpParseError("Frame too short to contain timestamp")

        obj = cls.__new__(cls)
        obj._bytes = raw
        obj._payload = _UNDECODED
        return obj

    @property
    def message_type(self) -> MessageType:
        """Return the MessageType of this message."""
        return MessageType(self._bytes[0] & 0x03)

    @property
    def has_error(self) -> bool:
        """Return True if the error flag is set in this message."""
        return bool(self._bytes[0] & 0x08)

    @property
    def address(self) -> int:
        """Return the address byte of this message."""
        return self._bytes[2]

    @property
    def port(self) -> int:
        """Return the port byte of this message."""
        return self._bytes[3]

    @property
    def payload_type(self) -> PayloadType:
        """Return the PayloadType of this message."""
        return decode_payload_type(self._bytes[4]).payload_type

    @property
    def has_timestamp(self) -> bool:
        """Return True if the timestamp flag is set in this message."""
        return bool(self._bytes[4] & _TIMESTAMP_FLAG)

    @property
    def timestamp(self) -> float | None:
        """Return the timestamp of this message, or None if not present."""
        if not self.has_timestamp:
            return None
        seconds, microseconds = struct.unpack_from("<IH", self._bytes, _HEADER_LEN)
        return cast(int, seconds) + cast(int, microseconds) * _TICK_PERIOD_S

    @property
    def payload_bytes(self) -> memoryview:
        """Payload bytes, excluding timestamp and checksum."""
        offset = _TIMESTAMPED_PAYLOAD_OFFSET if self.has_timestamp else _HEADER_LEN
        return memoryview(self._bytes)[offset:-1]

    @property
    def has_payload(self) -> bool:
        """Return True if a register has decoded the payload of this message."""
        return self._payload is not _UNDECODED

    @property
    def payload(self) -> P:
        """The decoded payload, as the register that parsed this message defines it.

        Only a register knows which contract a frame satisfies, so a message read from
        the wire carries no payload until one decodes it. Raises ``ValueError`` in that
        case; test with ``has_payload`` first, or read ``payload_bytes`` instead.
        """
        if self._payload is _UNDECODED:
            raise ValueError(
                "No register has decoded this message, so it has no payload. "
                "Parse it with a register, or read payload_bytes instead."
            )
        return self._payload

    def decode(self, decoder: type[PayloadDecoder[_P]]) -> "HarpMessage[_P]":
        """Return a copy of this message with its payload decoded by ``decoder``.

        The payload is derived from the frame in the same call, so the two cannot
        disagree. The payload type and the byte count are both checked, since together
        they decide whether these bytes can be read as this payload at all. The address
        is not, so a frame may be decoded by anything describing the same layout.
        """
        if self.payload_type is not decoder.payload_type:
            raise HarpParseError(
                f"{decoder.__name__} declares {decoder.payload_type!r} but this "
                f"message declares {self.payload_type!r}."
            )
        expected = decoder.payload_class.payload_dtype.itemsize
        actual = len(self.payload_bytes)
        if actual != expected:
            raise HarpParseError(
                f"{decoder.__name__} reads {expected} payload bytes but this message "
                f"carries {actual}."
            )
        obj: HarpMessage[_P] = HarpMessage.__new__(HarpMessage)
        obj._bytes = self._bytes
        obj._payload = decoder.parse(self)
        return obj

    @property
    def bytes(self) -> bytes:
        """The complete raw message frame, including checksum."""
        return self._bytes

    def __str__(self) -> str:
        return (
            f"HarpMessage(message_type={self.message_type!r}, address={self.address:#04x}, "
            f"payload_type={self.payload_type!r}, timestamp={self.timestamp!r})"
        )
