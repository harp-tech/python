import struct
from typing import ClassVar

import numpy as np
import pytest
from harp.protocol._message import HarpMessage, HarpParseError
from harp.protocol._payload import Field, StructPayload
from harp.protocol._payload_converters import IdentityConverter
from harp.protocol._register import RegisterBase, RegisterU8, RegisterU16
from harp.protocol._message_type import MessageType
from harp.protocol._payload_type import PayloadType

from tests.fixtures import TIMESTAMP_1S, make_frame_from_raw


def test_parse_read_request():
    # Read request: no payload, no timestamp.
    frame = make_frame_from_raw(0x01, address=8, port=0xFF, payload_type=0x04, payload=b"")
    msg = HarpMessage.parse(frame)
    assert msg.message_type == MessageType.Read
    assert msg.has_error is False
    assert msg.address == 8
    assert msg.port == 0xFF
    assert msg.payload_bytes == b""
    assert msg.timestamp is None


def test_parse_write_u8_payload():
    frame = make_frame_from_raw(0x02, address=10, port=0xFF, payload_type=0x01, payload=b"\x05")
    msg = HarpMessage.parse(frame)
    assert msg.message_type == MessageType.Write
    assert msg.payload_bytes == b"\x05"
    assert msg.payload_type == PayloadType.U8


def test_parse_with_timestamp():
    frame = make_frame_from_raw(
        0x03,
        address=32,
        port=0xFF,
        payload_type=0x11,  # U8 + HasTimestamp
        payload=b"\x7f",
        timestamp=TIMESTAMP_1S,
    )
    msg = HarpMessage.parse(frame)
    assert msg.message_type == MessageType.Event
    assert msg.timestamp == pytest.approx(1.0)
    assert msg.payload_bytes == b"\x7f"


def test_parse_error_flag():
    frame = make_frame_from_raw(0x09, address=0, port=0xFF, payload_type=0x01, payload=b"\x00")
    msg = HarpMessage.parse(frame)
    assert msg.has_error is True
    assert msg.message_type == MessageType.Read


def test_parse_u16_array():
    payload = struct.pack("<HHH", 100, 200, 300)
    frame = make_frame_from_raw(0x03, address=32, port=0xFF, payload_type=0x02, payload=payload)
    msg = HarpMessage.parse(frame)
    arr = np.frombuffer(msg.payload_bytes, dtype=np.dtype("<u2"))
    assert list(arr) == [100, 200, 300]


def test_parse_bad_checksum():
    frame = bytearray(make_frame_from_raw(0x01, 8, 0xFF, 0x01, b""))
    frame[-1] ^= 0xFF  # corrupt checksum
    with pytest.raises(HarpParseError, match="[Cc]hecksum"):
        HarpMessage.parse(bytes(frame))


def test_parse_too_short():
    with pytest.raises(HarpParseError):
        HarpMessage.parse(b"\x01\x04\x00")  # truncated


def test_parse_bad_payload_type():
    frame = make_frame_from_raw(0x01, 8, 0xFF, 0x00, b"")  # size nibble=0 invalid
    # Recompute checksum after we force a bad payload_type byte.
    # make_frame uses the passed payload_type directly, but 0x00 has size=0 which
    # is invalid. The frame checksum is still correct; parse should reject the type.
    with pytest.raises(HarpParseError):
        HarpMessage.parse(frame)


def test_parse_length_mismatch():
    # Build a frame then lie about the length byte.
    frame = bytearray(make_frame_from_raw(0x01, 8, 0xFF, 0x01, b""))
    frame[1] = 99  # wrong length
    # Recompute checksum to make it pass checksum check, still fails length check.
    frame[-1] = sum(frame[:-1]) & 0xFF
    with pytest.raises(HarpParseError):
        HarpMessage.parse(bytes(frame))


def test_wire_message_carries_no_payload():
    # A frame declares how its payload is encoded, not which register contract it
    # satisfies, so only a register can supply one.
    msg = HarpMessage.parse(
        make_frame_from_raw(0x02, address=10, port=0xFF, payload_type=0x01, payload=b"\x05")
    )
    assert msg.has_payload is False
    assert msg.payload_bytes == b"\x05"
    with pytest.raises(ValueError, match="has no payload"):
        msg.payload


def _u8_frame():
    return HarpMessage.parse(
        make_frame_from_raw(0x02, address=10, port=0xFF, payload_type=0x01, payload=b"\x05")
    )


def test_decode_attaches_payload_over_same_frame():
    typed = _u8_frame().decode(RegisterU8(0x0A))
    assert typed.has_payload is True
    assert typed.payload == 5
    assert typed.payload_bytes == b"\x05"
    assert typed.address == 10


def test_decode_leaves_source_undecoded():
    # The dispatch loop hands one frame to several places, so decoding must not mutate it.
    msg = _u8_frame()
    msg.decode(RegisterU8(0x0A))
    assert msg.has_payload is False


def test_decode_rejects_payload_type_mismatch():
    # Without the check the register reads the low bytes at its own width and returns a
    # silently wrong value, which is what parse does on its own.
    with pytest.raises(HarpParseError, match="declares"):
        _u8_frame().decode(RegisterU16(0x0A))


def test_decode_accepts_register_at_another_address():
    # The payload type decides whether these bytes can be read as this register at all.
    # The address says which register the device meant, so an identical layout decodes
    # either way and a frame can be read through more than one register.
    assert _u8_frame().decode(RegisterU8(0x2A)).payload == 5


class _SettingsPayload(StructPayload[np.uint16], length=3):
    gain: np.uint16 = Field(IdentityConverter(np.uint16))
    offset: np.uint16 = Field(IdentityConverter(np.uint16), offset=1)
    threshold: np.uint16 = Field(IdentityConverter(np.uint16), offset=2)


class _Settings(RegisterBase[_SettingsPayload]):
    address: ClassVar[int] = 32
    payload_type: ClassVar[PayloadType] = PayloadType.U16
    payload_class: ClassVar = _SettingsPayload


def test_decode_accepts_payload_spanning_several_elements():
    # decode sizes the payload from the payload class, as parse does, so several elements
    # are not read as one.
    payload = _SettingsPayload(gain=np.uint16(7), offset=np.uint16(8), threshold=np.uint16(9))
    typed = HarpMessage.parse(_Settings.format(payload)).decode(_Settings)
    assert (int(typed.payload.gain), int(typed.payload.threshold)) == (7, 9)


def test_decode_rejects_payload_of_wrong_byte_count():
    # A read request carries no payload, so the count check stops a register reading
    # past the frame.
    request = HarpMessage.parse(_Settings.format(message_type=MessageType.Read))
    with pytest.raises(HarpParseError, match="reads 6 payload bytes but this message carries 0"):
        request.decode(_Settings)
