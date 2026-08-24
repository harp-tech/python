import struct

from harp.device.client._framer import HarpFramer
from harp.protocol._message_type import MessageType

from tests.fixtures import TIMESTAMP_1S, make_frame_from_raw


def test_single_message():
    frame = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x01")
    msgs = HarpFramer.parse_bytes(frame)
    assert len(msgs) == 1
    assert msgs[0].address == 10
    assert msgs[0].payload_bytes == b"\x01"


def test_back_to_back_messages():
    f1 = make_frame_from_raw(0x01, 8, 0xFF, 0x04, b"")
    f2 = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x05")
    f3 = make_frame_from_raw(0x03, 32, 0xFF, 0x11, b"\x7f", timestamp=TIMESTAMP_1S)
    msgs = HarpFramer.parse_bytes(f1 + f2 + f3)
    assert len(msgs) == 3
    assert msgs[0].message_type == MessageType.Read
    assert msgs[1].message_type == MessageType.Write
    assert msgs[2].message_type == MessageType.Event


def test_garbage_prefix_skipped():
    garbage = bytes([0x00, 0x05, 0xFF, 0x20, 0x00])
    frame = make_frame_from_raw(0x01, 8, 0xFF, 0x04, b"")
    msgs = HarpFramer.parse_bytes(garbage + frame)
    assert len(msgs) == 1
    assert msgs[0].address == 8


def test_garbage_between_messages():
    f1 = make_frame_from_raw(0x01, 8, 0xFF, 0x04, b"")
    f2 = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x05")
    noise = bytes([0xAA, 0xBB, 0xCC])
    msgs = HarpFramer.parse_bytes(f1 + noise + f2)
    assert len(msgs) == 2


def test_bad_checksum_skipped_recovery():
    # A frame with a bad checksum should be skipped; the next valid frame parses.
    bad = bytearray(make_frame_from_raw(0x01, 8, 0xFF, 0x04, b""))
    bad[-1] ^= 0xFF  # corrupt checksum
    good = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x05")
    msgs = HarpFramer.parse_bytes(bytes(bad) + good)
    assert len(msgs) == 1
    assert msgs[0].address == 10


def test_truncated_stream_returns_empty():
    frame = make_frame_from_raw(0x01, 8, 0xFF, 0x04, b"")
    # Only the first 3 bytes, not enough for a complete frame.
    msgs = HarpFramer.parse_bytes(frame[:3])
    assert msgs == []


def test_incremental_feed():
    # Feeding data in small chunks still yields the complete message.
    frame = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x42")
    framer = HarpFramer()
    results = []
    for byte in frame:
        framer.feed(bytes([byte]))
        results.extend(framer.frames())
    assert len(results) == 1
    assert results[0].payload_bytes == b"\x42"


def test_all_scalar_types():
    # Framer correctly parses messages with each PayloadType.
    from harp.protocol._payload_type import PayloadType, encode_payload_type

    for pt in PayloadType:
        size = pt.numpy_dtype.itemsize
        payload = bytes(range(size))
        pt_byte = encode_payload_type(pt)
        frame = make_frame_from_raw(0x03, 32, 0xFF, pt_byte, payload)
        msgs = HarpFramer.parse_bytes(frame)
        assert len(msgs) == 1, f"Failed for {pt}"
        assert msgs[0].payload_type == pt


def test_array_payload():
    payload = struct.pack("<" + "H" * 5, *range(5))
    frame = make_frame_from_raw(0x03, 32, 0xFF, 0x02, payload)
    msgs = HarpFramer.parse_bytes(frame)
    assert len(msgs) == 1
    assert len(msgs[0].payload_bytes) == 10


def test_parse_file(tmp_path):
    frame = make_frame_from_raw(0x01, 8, 0xFF, 0x04, b"")
    p = tmp_path / "test.bin"
    p.write_bytes(frame)
    msgs = HarpFramer.parse_file(p)
    assert len(msgs) == 1
    assert len(msgs) == 1
